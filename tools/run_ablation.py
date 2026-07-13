"""Ablation study runner for proposal method comparison.

Compares BEV Connected Components vs DBSCAN variants on detection
metrics and performance.

Usage:
    uv run python tools/run_ablation.py --config configs/ablation/proposal_cmp.yaml
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io.boxes_to_labels import BoxLabelMapping
from src.utils.config import load_config


@dataclass
class AblationResult:
    """Results for a single ablation method."""

    method: str
    description: str
    metrics: dict[str, float]
    distance_metrics: dict[str, dict[str, float]]
    timing: dict[str, dict[str, float]]
    num_frames: int


def run_method(
    method_name: str,
    config: dict,
    dataset,
    labels_helper,
    frame_range: range,
) -> AblationResult:
    """Run a single proposal method and collect metrics."""
    from src.boxes.obb import fit_obbs_to_clusters
    from src.eval.detection_eval import (
        BoundingBox3D,
        DetectionEvaluator,
        extract_gt_instances,
    )
    from src.preprocess.ground import segment_ground
    from src.preprocess.roi import ROIBounds, crop_points
    from src.preprocess.voxel import voxel_downsample_fixed
    from src.proposals.bev_cc import cluster_bev_cc
    from src.proposals.dbscan import (
        cluster_dbscan_fixed,
        cluster_dbscan_distance_adaptive,
    )
    from src.proposals.filtering import filter_clusters

    print(f"\n{'=' * 60}")
    print(f"Running method: {method_name}")
    print(f"{'=' * 60}")

    # Initialize evaluator
    evaluation_cfg = config.get("evaluation", {})
    evaluator = DetectionEvaluator(
        iou_threshold=evaluation_cfg.get("iou_threshold", 0.5),
        use_3d_iou=evaluation_cfg.get("use_3d_iou", False),
        distance_bins=evaluation_cfg.get("distance_bins"),
    )

    preprocess_cfg = config.get("preprocess", {})
    roi_cfg = preprocess_cfg.get("roi", {})
    voxel_cfg = preprocess_cfg.get("voxel", {})
    ground_cfg = config.get("ground", {})
    filter_cfg = config.get("filtering", {})
    box_cfg = config.get("boxes", {})
    bev_cfg = config.get("bev_cc", {})
    dbscan_fixed_cfg = config.get("dbscan_fixed", {})
    dbscan_adaptive_cfg = config.get("dbscan_adaptive", {})

    # Timing statistics
    timing_data: dict[str, list[float]] = {
        "preprocess": [],
        "ground": [],
        "proposals": [],
        "filtering": [],
        "boxes": [],
        "total": [],
    }

    num_proposals_list: list[int] = []
    class_mapping = BoxLabelMapping()

    for frame_idx in frame_range:
        frame_start = time.perf_counter()

        # Load frame
        frame = dataset.get_frame(frame_idx)
        points = frame.points

        # Preprocess
        t0 = time.perf_counter()
        roi_bounds = ROIBounds(
            x_min=roi_cfg.get("x_min", 0),
            x_max=roi_cfg.get("x_max", 70),
            y_min=roi_cfg.get("y_min", -40),
            y_max=roi_cfg.get("y_max", 40),
            z_min=roi_cfg.get("z_min", -3),
            z_max=roi_cfg.get("z_max", 3),
        )
        roi_result = crop_points(points, roi_bounds)
        points = roi_result.points
        if voxel_cfg.get("enabled", True):
            voxel_result = voxel_downsample_fixed(
                points, voxel_cfg.get("voxel_size", 0.1)
            )
            points = voxel_result.points
        timing_data["preprocess"].append((time.perf_counter() - t0) * 1000)

        # Ground segmentation
        t0 = time.perf_counter()
        ground_result = segment_ground(
            points,
            distance_threshold=ground_cfg.get("distance_threshold", 0.2),
            ransac_n=ground_cfg.get("ransac_n", 3),
            num_iterations=ground_cfg.get("num_iterations", 1000),
            z_min=ground_cfg.get("z_min", -2.5),
            z_max=ground_cfg.get("z_max", -0.5),
            min_normal_z=ground_cfg.get("min_normal_z", 0.9),
        )
        non_ground_points = ground_result.nonground_points
        timing_data["ground"].append((time.perf_counter() - t0) * 1000)

        # Proposal generation (method-specific)
        t0 = time.perf_counter()
        if method_name == "bev_cc":
            proposal_result = cluster_bev_cc(
                non_ground_points,
                resolution=bev_cfg.get("resolution", 0.2),
                min_height=bev_cfg.get("min_height", 0.3),
                min_points=bev_cfg.get("min_points", 10),
            )
        elif method_name == "dbscan_fixed":
            proposal_result = cluster_dbscan_fixed(
                non_ground_points,
                eps=dbscan_fixed_cfg.get("eps", 0.5),
                min_points=dbscan_fixed_cfg.get("min_points", 10),
            )
        elif method_name == "dbscan_adaptive":
            # Use proper distance-adaptive DBSCAN per milestone 4.2
            proposal_result = cluster_dbscan_distance_adaptive(
                non_ground_points,
                distance_bins=dbscan_adaptive_cfg.get("distance_bins", [0, 20, 40, 60]),
                bin_eps=dbscan_adaptive_cfg.get("bin_eps", [0.3, 0.5, 0.8]),
                min_points=dbscan_adaptive_cfg.get("min_points", 10),
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")
        clusters = proposal_result.clusters
        timing_data["proposals"].append((time.perf_counter() - t0) * 1000)

        # Filter clusters
        t0 = time.perf_counter()
        filter_result = filter_clusters(
            non_ground_points,
            clusters,
            config=filter_cfg,
        )
        filtered_clusters = filter_result.clusters
        timing_data["filtering"].append((time.perf_counter() - t0) * 1000)

        # Fit bounding boxes
        t0 = time.perf_counter()
        box_results = fit_obbs_to_clusters(
            non_ground_points,
            filtered_clusters,
            z_percentile=box_cfg.get("z_percentile", 5.0),
            min_volume=box_cfg.get("min_box_volume", 0.1),
            max_volume=box_cfg.get("max_box_volume", 1000),
        )
        timing_data["boxes"].append((time.perf_counter() - t0) * 1000)

        # Convert to evaluation format
        pred_boxes = []
        for br in box_results:
            if br.valid and br.params is not None:
                box_record = {
                    "dimensions": list(br.params.dimensions),
                    "volume": br.params.volume,
                }
                pred_boxes.append(
                    BoundingBox3D(
                        center=br.params.center.copy(),
                        dimensions=np.array(br.params.dimensions, dtype=np.float64),
                        yaw=br.params.yaw,
                        semantic_id=class_mapping.infer_class_from_box(box_record),
                        score=1.0,
                        num_points=br.num_points,
                    )
                )

        num_proposals_list.append(len(pred_boxes))

        # Get ground truth
        if frame.inst_label is not None:
            gt_boxes = extract_gt_instances(
                frame.points,
                frame.inst_label,
                frame.sem_label,
                labels_helper,
            )
            evaluator.add_frame(pred_boxes, gt_boxes, frame_idx)

        timing_data["total"].append((time.perf_counter() - frame_start) * 1000)

        if (frame_idx + 1) % 50 == 0:
            print(f"  Processed frame {frame_idx + 1}")

    # Compute metrics
    metrics = evaluator.compute_metrics()
    distance_metrics = evaluator.compute_distance_metrics()

    # Compute timing statistics
    timing_stats = {}
    for key, values in timing_data.items():
        if not values:
            continue
        arr = np.array(values)
        timing_stats[key] = {
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p99_ms": float(np.percentile(arr, 99)),
        }

    # Method description
    method_info = next(
        (
            m
            for m in config.get("ablation", {}).get("methods", [])
            if m["name"] == method_name
        ),
        {"description": method_name},
    )

    return AblationResult(
        method=method_name,
        description=method_info.get("description", method_name),
        metrics={
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "mean_iou": metrics.mean_iou,
            "num_proposals_avg": float(np.mean(num_proposals_list)),
        },
        distance_metrics={
            k: {"precision": v.precision, "recall": v.recall, "f1": v.f1}
            for k, v in distance_metrics.items()
        },
        timing=timing_stats,
        num_frames=len(frame_range),
    )


def generate_report(results: list[AblationResult], output_dir: Path) -> None:
    """Generate ablation study report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON results
    json_path = output_dir / "ablation_proposals.json"
    json_data = {
        "results": [
            {
                "method": r.method,
                "description": r.description,
                "metrics": r.metrics,
                "distance_metrics": r.distance_metrics,
                "timing": r.timing,
                "num_frames": r.num_frames,
            }
            for r in results
        ]
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved JSON results to {json_path}")

    # Markdown report
    md_path = output_dir / "ablation_proposals.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Ablation Study: Proposal Method Comparison\n\n")

        f.write("## Summary\n\n")
        f.write(
            "| Method | Precision | Recall | F1 | Mean IoU | Avg Proposals | Avg Time (ms) |\n"
        )
        f.write(
            "|--------|-----------|--------|----|-----------|--------------|--------------|\n"
        )
        for r in results:
            f.write(
                f"| {r.description} | {r.metrics['precision']:.3f} | "
                f"{r.metrics['recall']:.3f} | {r.metrics['f1']:.3f} | "
                f"{r.metrics['mean_iou']:.3f} | {r.metrics['num_proposals_avg']:.1f} | "
                f"{r.timing.get('total', {}).get('mean_ms', 0):.1f} |\n"
            )

        f.write("\n## Distance-Stratified F1 Scores\n\n")
        f.write("| Method | 0-10m | 10-20m | 20-30m | 30-40m | 40-50m |\n")
        f.write("|--------|-------|--------|--------|--------|--------|\n")
        for r in results:
            f.write(f"| {r.description} ")
            for dist_key in ["0-10", "10-20", "20-30", "30-40", "40-50"]:
                dm = r.distance_metrics.get(f"{dist_key}m", {})
                f1 = dm.get("f1", 0)
                f.write(f"| {f1:.3f} ")
            f.write("|\n")

        f.write("\n## Timing Breakdown (P50)\n\n")
        f.write(
            "| Method | Preprocess | Ground | Proposals | Filtering | Boxes | Total |\n"
        )
        f.write(
            "|--------|------------|--------|-----------|-----------|-------|-------|\n"
        )
        for r in results:
            f.write(f"| {r.description} ")
            for key in [
                "preprocess",
                "ground",
                "proposals",
                "filtering",
                "boxes",
                "total",
            ]:
                p50 = r.timing.get(key, {}).get("p50_ms", 0)
                f.write(f"| {p50:.1f} ")
            f.write("|\n")

        f.write("\n## Conclusions\n\n")
        f.write("*TODO: Add analysis based on results*\n")

    print(f"Saved Markdown report to {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ablation/proposal_cmp.yaml"),
        help="Path to ablation config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports"),
        help="Output directory",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Ablation Study: Proposal Method Comparison")
    print("=" * 60)

    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Import modules
    from src.datasets.kitti_odometry import KITTIDataset
    from src.datasets.semkitti_labels import SemanticKITTILabels

    # Load dataset
    dataset_cfg = config.get("dataset", {})
    dataset_root = Path(dataset_cfg.get("root", "."))
    sequence = dataset_cfg.get("sequence", "00")

    print(f"Loading dataset: {dataset_root}, sequence {sequence}")
    dataset = KITTIDataset(dataset_root, sequence)
    labels_helper = SemanticKITTILabels()

    # Frame range
    frame_cfg = dataset_cfg.get("frame_range", {})
    start = frame_cfg.get("start", 0)
    end = min(frame_cfg.get("end", dataset.num_frames), dataset.num_frames)
    step = frame_cfg.get("step", 1)
    frame_range = range(start, end, step)
    print(
        f"Processing frames {start} to {end}, step {step} ({len(frame_range)} frames)"
    )

    # Run methods
    methods = config.get("ablation", {}).get("methods", [])
    results: list[AblationResult] = []

    for method_info in methods:
        method_name = method_info["name"]
        try:
            result = run_method(
                method_name, config, dataset, labels_helper, frame_range
            )
            results.append(result)
        except Exception as e:
            print(f"Error running method {method_name}: {e}")
            import traceback

            traceback.print_exc()

    # Generate report
    generate_report(results, args.output)

    print("\n" + "=" * 60)
    print("Ablation study complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
