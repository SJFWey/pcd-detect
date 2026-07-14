"""
Eval command - Evaluate detection results.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import typer

from ..datasets.kitti_odometry import KITTIDataset
from ..datasets.semkitti_labels import SemanticKITTILabels
from ..eval.detection_eval import (
    BoundingBox3D,
    DetectionEvaluator,
    canonical_semantic_id,
    extract_reference_instances,
)
from ..io.boxes_to_labels import BoxLabelMapping
from ..io.export_boxes import JSONLBoxReader
from ..preprocess.roi import ROIBounds, apply_mask, roi_mask
from .common import load_config, normalize_target_name


TARGET_CLASS_IDS = {
    "car": {10, 252},
    "person": {30, 254},
    "bicyclist": {31, 253},
    "bicycle": {11},
    "motorcycle": {15, 255},
    "motorcyclist": {32, 255},
    "bus": {13, 257},
    "truck": {18, 258},
    "other_vehicle": {20, 259},
}



def _resolve_roi_bounds(config: dict) -> ROIBounds:
    roi_cfg = config.get("preprocess", {}).get("roi", {})
    return ROIBounds(
        x_min=float(roi_cfg.get("x_min", 0)),
        x_max=float(roi_cfg.get("x_max", 70)),
        y_min=float(roi_cfg.get("y_min", -40)),
        y_max=float(roi_cfg.get("y_max", 40)),
        z_min=float(roi_cfg.get("z_min", -3)),
        z_max=float(roi_cfg.get("z_max", 3)),
    )

def _normalize_class_map(
    per_class_iou: dict[str, float],
) -> tuple[dict[str, float], dict[str, str]]:
    """Normalize class name keys and keep original names."""
    norm_values: dict[str, float] = {}
    norm_names: dict[str, str] = {}
    for name, value in per_class_iou.items():
        key = normalize_target_name(str(name))
        if key not in norm_values:
            norm_values[key] = float(value)
            norm_names[key] = str(name)
    return norm_values, norm_names


def _resolve_eval_targets(cfg: dict) -> list[str]:
    """Resolve enabled evaluation targets from config."""
    eval_cfg = cfg.get("evaluation", {})
    targets = eval_cfg.get("targets") or eval_cfg.get("target_classes")
    if targets is None:
        targets = ["car", "person", "bicyclist"]

    if isinstance(targets, dict):
        names = [k for k, v in targets.items() if bool(v)]
    elif isinstance(targets, (list, tuple, set)):
        names = list(targets)
    elif isinstance(targets, str):
        names = [targets]
    else:
        names = []

    resolved: list[str] = []
    seen = set()
    for name in names:
        if name is None:
            continue
        key = normalize_target_name(str(name))
        if key and key not in seen:
            resolved.append(key)
            seen.add(key)

    return resolved


def _ids_match_target(semantic_id: int, target_ids: set[int]) -> bool:
    """Check if a semantic id belongs to the enabled target set."""
    return int(semantic_id) in target_ids or canonical_semantic_id(semantic_id) in {
        canonical_semantic_id(target_id) for target_id in target_ids
    }


def _resolve_detection_target_ids(
    cfg: dict,
    labels_helper: SemanticKITTILabels,
) -> tuple[list[str], set[int], dict[str, set[int]]]:
    """Resolve evaluation target names to SemanticKITTI semantic ids."""
    targets = _resolve_eval_targets(cfg)
    if not targets:
        raise typer.BadParameter("evaluation.targets must enable at least one class")

    target_ids: set[int] = set()
    target_name_to_ids: dict[str, set[int]] = {}

    for target in targets:
        key = normalize_target_name(target)
        ids = set(TARGET_CLASS_IDS.get(key, set()))
        if not ids:
            ids = {
                int(label_id)
                for label_id, label_name in labels_helper.labels.items()
                if normalize_target_name(str(label_name)) == key
            }
        if not ids:
            raise typer.BadParameter(f"Unknown evaluation target class: {target}")

        target_ids.update(ids)
        target_name_to_ids[key] = ids

    return targets, target_ids, target_name_to_ids


def _semantic_id_from_box_record(
    box: dict[str, Any],
    mapping: BoxLabelMapping,
    target_name_to_ids: dict[str, set[int]],
) -> tuple[int, bool]:
    """Read an explicit prediction class or infer one from box dimensions."""
    for key in ("semantic_id", "class_id"):
        value = box.get(key)
        if value is not None:
            semantic_id = int(value)
            if semantic_id > 0:
                return semantic_id, False

    for key in ("class_name", "semantic_class", "label"):
        value = box.get(key)
        if value:
            class_key = normalize_target_name(str(value))
            ids = target_name_to_ids.get(class_key)
            if ids:
                return min(ids), False

    return mapping.infer_class_from_box(box), True


def _filter_per_class_iou(
    per_class_iou: dict[str, float],
    targets: list[str],
    name_map: dict[str, str] | None = None,
) -> tuple[dict[str, float], float | None, list[str]]:
    """Filter per-class IoU to target classes and compute mean."""
    norm_values, norm_names = _normalize_class_map(per_class_iou)
    missing: list[str] = []
    filtered: dict[str, float] = {}
    values: list[float] = []

    for target in targets:
        key = normalize_target_name(target)
        if key not in norm_values:
            missing.append(target)
            continue
        out_name = (name_map or norm_names).get(key, key)
        value = norm_values[key]
        filtered[out_name] = value
        values.append(value)

    mean = float(sum(values) / len(values)) if values else None
    return filtered, mean, missing


def _write_official_subset_report(
    cfg: dict, output_dir: Path, full_report: Path
) -> Path | None:
    """Write a subset report derived from the full official evaluation."""
    targets = _resolve_eval_targets(cfg)
    if not targets:
        typer.echo("No evaluation targets enabled in config; skipping subset report.")
        return None

    if not full_report.exists():
        typer.echo(f"Warning: Full official report not found: {full_report}")
        return None

    with open(full_report, "r", encoding="utf-8") as f:
        full = json.load(f)

    subset: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "source_report": str(full_report),
        "targets": targets,
        "dataset": full.get("dataset"),
        "predictions": full.get("predictions"),
        "split": full.get("split"),
        "backend": full.get("backend"),
    }

    sem = full.get("semantic_eval")
    name_map: dict[str, str] | None = None
    if sem and isinstance(sem, dict):
        metrics = sem.get("metrics", {}) or {}
        per_class = metrics.get("per_class_iou", {}) or {}
        norm_values, norm_names = _normalize_class_map(per_class)
        name_map = norm_names
        filtered, mean_iou, missing = _filter_per_class_iou(
            per_class, targets, name_map
        )
        if missing:
            typer.echo(
                "Warning: targets missing from full evaluation: "
                + ", ".join(sorted({normalize_target_name(m) for m in missing}))
            )

        subset["semantic_eval"] = {
            "success": sem.get("success"),
            "returncode": sem.get("returncode"),
            "metrics": {
                "per_class_iou": filtered,
                "mean_iou": mean_iou,
                "accuracy": None,
            },
        }

    dist = full.get("distance_eval")
    if dist and isinstance(dist, dict):
        dist_metrics = dist.get("metrics", {}) or {}
        bins = dist_metrics.get("distance_bins", {}) or {}
        filtered_bins: dict[str, Any] = {}
        for bin_name, m in bins.items():
            per_class = (m or {}).get("per_class_iou", {}) or {}
            filtered, mean_iou, _ = _filter_per_class_iou(per_class, targets, name_map)
            filtered_bins[bin_name] = {
                "per_class_iou": filtered,
                "mean_iou": mean_iou,
                "accuracy": None,
            }

        subset["distance_eval"] = {
            "success": dist.get("success"),
            "returncode": dist.get("returncode"),
            "metrics": {"distance_bins": filtered_bins},
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "official_metrics_targets.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)

    typer.echo(f"Subset report saved to: {output_path}")
    return output_path


def _generate_evaluation_curves(evaluator, output_dir: Path) -> None:
    """Generate evaluation curves (PR curve, distance F1 bar chart)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Distance-stratified F1 bar chart
        distance_metrics = evaluator.compute_distance_metrics()
        distances = []
        f1_scores = []
        for dist_bin, dm in distance_metrics.items():
            distances.append(dist_bin.replace("m", ""))
            f1_scores.append(dm.f1)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(distances, f1_scores, color="steelblue", edgecolor="black")
        ax.set_xlabel("Distance Range (m)", fontsize=12)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_title("Detection F1 by Distance", fontsize=14)
        ax.set_ylim(0, 1.0)

        # Add value labels on bars
        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            ax.annotate(
                f"{f1:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(figures_dir / "f1_by_distance.png", dpi=150)
        plt.close()

        # IoU histogram
        metrics = evaluator.compute_metrics()
        if metrics.iou_histogram:
            fig, ax = plt.subplots(figsize=(8, 6))
            bins = list(metrics.iou_histogram.keys())
            counts = list(metrics.iou_histogram.values())
            ax.bar(bins, counts, color="coral", edgecolor="black")
            ax.set_xlabel("IoU Range", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title("IoU Distribution of Matched Boxes", fontsize=14)
            plt.tight_layout()
            plt.savefig(figures_dir / "iou_histogram.png", dpi=150)
            plt.close()

        typer.echo(f"  Saved evaluation curves to {figures_dir}")

    except ImportError:
        typer.echo("  (matplotlib not available, skipping curve generation)")
    except Exception as e:
        typer.echo(f"  Warning: Failed to generate curves: {e}")


def _resolve_eval_frame_range(cfg: dict[str, Any], num_frames: int) -> range:
    """Resolve evaluation frames from the shared frames config."""
    frames_cfg = cfg.get("frames", {})
    start_raw = frames_cfg.get("start", 0)
    end_raw = frames_cfg.get("end")
    step_raw = frames_cfg.get("step", 1)

    start = 0 if start_raw is None else int(start_raw)
    end = num_frames if end_raw is None else min(int(end_raw), num_frames)
    step = 1 if step_raw is None else int(step_raw)

    if start < 0:
        raise typer.BadParameter("frames.start must be >= 0")
    if start >= num_frames:
        raise typer.BadParameter(
            f"frames.start ({start}) out of range (0-{num_frames - 1})"
        )
    if end <= start:
        raise typer.BadParameter("frames.end must be > frames.start")
    if step <= 0:
        raise typer.BadParameter("frames.step must be >= 1")

    return range(start, end, step)


def eval_command(config: Path | None = None) -> None:
    """
    Evaluate detection results.

    All parameters are configured via configs/evaluation.yaml.
    Use --config to override with a custom config file.
    """
    cfg = load_config(config, task="eval")

    # Get evaluation settings from config
    eval_cfg = cfg.get("evaluation", {})
    official = eval_cfg.get("official", False)
    detection = eval_cfg.get("detection", True)
    split = eval_cfg.get("split", "valid")
    iou_threshold = eval_cfg.get("iou_threshold", 0.5)

    # Determine paths
    predictions_path = eval_cfg.get("predictions_path")
    if predictions_path:
        predictions = Path(predictions_path)
        if not predictions.is_absolute():
            predictions = Path(cfg["output"]["root"]) / predictions
    else:
        predictions = Path(cfg["output"]["root"]) / cfg["output"]["predictions_dir"]
    output_dir = Path(cfg["output"]["root"]) / cfg["output"]["reports_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("=" * 60)
    typer.echo("Evaluation")
    typer.echo("=" * 60)
    typer.echo(f"Dataset: {cfg['dataset']['root']}")
    typer.echo(f"Predictions: {predictions}")
    typer.echo(f"Split: {split}")
    typer.echo(f"Output: {output_dir}")
    typer.echo()

    # If no specific evaluation is requested, show help
    if not (official or detection):
        typer.echo("No evaluation type enabled in config. Set one or more of:")
        typer.echo("  evaluation.official: true")
        typer.echo("  evaluation.detection: true")
        return

    # Run official evaluation
    if official:
        typer.echo("-" * 40)
        typer.echo("Running Official SemanticKITTI Evaluation...")
        typer.echo("-" * 40)

        # Check if predictions exist
        pred_seq_dir = predictions / "sequences"
        if not pred_seq_dir.exists():
            typer.echo(f"Error: No predictions found at {predictions}")
            typer.echo()
            typer.echo(
                "The official SemanticKITTI evaluation requires per-point predictions"
            )
            typer.echo(
                "in the format: {predictions}/sequences/XX/predictions/XXXXXX.label"
            )
            typer.echo()
            typer.echo("To generate predictions from ground truth (for testing):")
            typer.echo("  1) Set export.type: predictions in configs/export.yaml")
            typer.echo("  2) Run: uv run python -m src.main export")
            raise typer.Exit(code=1)

        # Use the run_official_eval.py script (vendored wrapper).
        eval_script = (
            Path(__file__).parent.parent.parent / "tools" / "run_official_eval.py"
        )
        if eval_script.exists():
            repo_root = Path(__file__).parent.parent.parent
            datacfg = eval_cfg.get("datacfg")
            datacfg_path = None
            if datacfg:
                datacfg_path = Path(datacfg)
                if not datacfg_path.is_absolute():
                    datacfg_path = repo_root / datacfg_path

            full_cmd = [
                sys.executable,
                str(eval_script),
                "--dataset",
                str(cfg["dataset"]["root"]),
                "--predictions",
                str(predictions),
                "--split",
                split,
                "--output",
                str(output_dir / "official_metrics.json"),
            ]
            if datacfg_path is not None:
                full_cmd.extend(["--datacfg", str(datacfg_path)])
            typer.echo(f"Running (full classes): {' '.join(full_cmd)}")
            full_result = subprocess.run(full_cmd)

            if full_result.returncode == 0:
                _write_official_subset_report(
                    cfg, output_dir, output_dir / "official_metrics.json"
                )

            else:
                typer.echo(
                    "Error: Official evaluation returned non-zero exit code "
                    f"{full_result.returncode}"
                )
                raise typer.Exit(code=full_result.returncode or 1)
        else:
            typer.echo(f"Error: Evaluation script not found: {eval_script}")
            raise typer.Exit(code=1)
        typer.echo()

    # Run detection evaluation
    if detection:
        typer.echo("-" * 40)
        typer.echo("Running Proxy Detection Metrics Evaluation...")
        typer.echo("-" * 40)

        try:
            # Load dataset
            dataset_root = Path(cfg["dataset"]["root"])
            sequence = cfg["dataset"]["sequence"]

            typer.echo(f"Loading dataset sequence {sequence}...")
            kitti_dataset = KITTIDataset(dataset_root, sequence)
            labels_helper = SemanticKITTILabels()
            targets, target_ids, target_name_to_ids = _resolve_detection_target_ids(
                cfg, labels_helper
            )
            frame_range = _resolve_eval_frame_range(cfg, kitti_dataset.num_frames)
            class_mapping = BoxLabelMapping()
            typer.echo(
                "Detection targets: "
                + ", ".join(targets)
                + f" (semantic ids: {sorted(target_ids)})"
            )
            typer.echo(
                f"Evaluation frames: {frame_range.start} to {frame_range.stop - 1}"
                + (f" (step {frame_range.step})" if frame_range.step != 1 else "")
            )
            typer.echo(
                "Prediction boxes without semantic class are inferred from box dimensions."
            )

            # Initialize evaluator
            use_3d_iou = bool(eval_cfg.get("use_3d_iou", False))
            distance_bins_raw = eval_cfg.get("distance_bins")
            distance_bins: list[float] | None = None
            if distance_bins_raw is not None:
                if not isinstance(distance_bins_raw, (list, tuple)):
                    raise typer.BadParameter(
                        "evaluation.distance_bins must be a list of numbers"
                    )
                try:
                    distance_bins = [float(x) for x in distance_bins_raw]
                except (TypeError, ValueError) as exc:
                    raise typer.BadParameter(
                        "evaluation.distance_bins must be a list of numbers"
                    ) from exc
                if len(distance_bins) < 2:
                    raise typer.BadParameter(
                        "evaluation.distance_bins must have at least 2 entries"
                    )

            evaluator = DetectionEvaluator(
                iou_threshold=iou_threshold,
                use_3d_iou=use_3d_iou,
                distance_bins=distance_bins,
                match_classes=True,
            )

            # Try to load predicted boxes
            boxes_dir = cfg["output"].get("boxes_dir", "boxes")
            boxes_path = (
                Path(cfg["output"]["root"]) / boxes_dir / sequence / "boxes.jsonl"
            )
            if not boxes_path.exists():
                # Try alternate path format
                boxes_path = Path(cfg["output"]["root"]) / f"boxes_{sequence}.jsonl"

            if boxes_path.exists():
                typer.echo(f"Loading predictions from {boxes_path}...")
                reader = JSONLBoxReader(boxes_path)
                evaluated_frame_count = 0
                skipped_label_frame_count = 0

                for i, frame_id in enumerate(frame_range, start=1):
                    # Fit proxy reference boxes from visible labeled points.
                    frame_data = kitti_dataset.get_frame(frame_id)
                    if frame_data.inst_label is None or frame_data.sem_label is None:
                        skipped_label_frame_count += 1
                        continue

                    roi_bounds = _resolve_roi_bounds(cfg)
                    gt_support_mask = roi_mask(frame_data.points, roi_bounds)
                    gt_points, gt_instance_labels, gt_semantic_labels = apply_mask(
                        gt_support_mask,
                        frame_data.points,
                        frame_data.inst_label,
                        frame_data.sem_label,
                    )
                    gt_boxes = extract_reference_instances(
                        gt_points,
                        gt_instance_labels,
                        gt_semantic_labels,
                        labels_helper,
                    )
                    gt_boxes = [
                        box
                        for box in gt_boxes
                        if _ids_match_target(box.semantic_id, target_ids)
                    ]

                    # Load predictions for this frame
                    pred_boxes_data = reader.read_frame(frame_id)
                    pred_boxes = []
                    inferred_classes = 0
                    if pred_boxes_data:
                        for box in pred_boxes_data.boxes:
                            if box.valid:
                                box_record = box.to_dict()
                                semantic_id, inferred = _semantic_id_from_box_record(
                                    box_record, class_mapping, target_name_to_ids
                                )
                                if not _ids_match_target(semantic_id, target_ids):
                                    continue
                                if inferred:
                                    inferred_classes += 1
                                pred_boxes.append(
                                    BoundingBox3D(
                                        center=np.array(box.center, dtype=np.float64),
                                        dimensions=np.array(
                                            box.dimensions, dtype=np.float64
                                        ),
                                        yaw=box.yaw,
                                        semantic_id=semantic_id,
                                        score=box.score,
                                        num_points=box.num_points,
                                    )
                                )
                    if inferred_classes and frame_id == 0:
                        typer.echo(
                            "  Inferred semantic classes for prediction boxes "
                            "using size heuristics."
                        )

                    # Add to evaluator
                    evaluator.add_frame(pred_boxes, gt_boxes, frame_id)
                    evaluated_frame_count += 1

                    # Progress
                    if i % 100 == 0:
                        typer.echo(
                            f"  Evaluated {i}/{len(frame_range)} frames..."
                        )

                if evaluated_frame_count == 0:
                    raise typer.BadParameter(
                        "No requested frames contain both instance and semantic labels."
                    )
                if skipped_label_frame_count:
                    typer.echo(
                        f"Skipped {skipped_label_frame_count} frame(s) without point labels."
                    )

                # Compute and display metrics
                metrics = evaluator.compute_metrics()
                typer.echo()
                typer.echo("Proxy Detection Metrics:")
                typer.echo(f"  Precision: {metrics.precision:.4f}")
                typer.echo(f"  Recall:    {metrics.recall:.4f}")
                typer.echo(f"  F1 Score:  {metrics.f1:.4f}")
                typer.echo(f"  Mean matched IoU: {metrics.mean_iou:.4f}")
                typer.echo()

                # Distance-stratified metrics
                distance_metrics = evaluator.compute_distance_metrics()
                typer.echo("Distance-Stratified Metrics:")
                for dist_bin, dm in distance_metrics.items():
                    typer.echo(
                        f"  {dist_bin}: P={dm.precision:.3f}, R={dm.recall:.3f}, F1={dm.f1:.3f}"
                    )

                # Save results
                evaluator.save_results(output_dir / "det_metrics.json")
                typer.echo(
                    f"\nSaved proxy detection metrics to {output_dir / 'det_metrics.json'}"
                )

                # Generate evaluation curves
                _generate_evaluation_curves(evaluator, output_dir)

            else:
                typer.echo(f"Error: No prediction boxes found at {boxes_path}")
                typer.echo("Run the detection pipeline first to generate predictions.")
                raise typer.Exit(code=1)

        except typer.Exit:
            raise
        except typer.BadParameter:
            raise
        except ImportError as exc:
            typer.echo(f"Import error: {exc}")
            typer.echo("Make sure all dependencies are installed.")
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            typer.echo(f"Error during detection evaluation: {exc}")
            raise typer.Exit(code=1) from exc

        typer.echo()

    typer.echo()
    typer.echo("=" * 60)
    typer.echo("Evaluation complete!")
    typer.echo("=" * 60)
