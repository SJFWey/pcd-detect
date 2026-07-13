#!/usr/bin/env python3
"""Run official SemanticKITTI evaluation pipeline.

This tool provides a unified interface to run both:
- evaluate_semantics.py: Overall semantic IoU metrics
- evaluate_semantics_by_distance.py: Distance-stratified IoU metrics

Results are saved to outputs/reports/official_metrics.json.

Usage:
    # Run both evaluations on validation split
    uv run python tools/run_official_eval.py \\
        --dataset /path/to/kitti/dataset \\
        --predictions outputs/predictions \\
        --split valid

    # Run only distance-stratified evaluation
    uv run python tools/run_official_eval.py \\
        --dataset /path/to/kitti/dataset \\
        --predictions outputs/predictions \\
        --split valid \\
        --distance-only

    # Custom output location
    uv run python tools/run_official_eval.py \\
        --dataset /path/to/kitti/dataset \\
        --predictions outputs/predictions \\
        --output outputs/reports/my_eval.json
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def get_third_party_path() -> Path:
    """Get path to third_party/semkitti_api directory."""
    return Path(__file__).resolve().parent.parent / "third_party" / "semkitti_api"


def get_script_path(script_name: str) -> Path:
    """Get path to an official SemanticKITTI API script."""
    path = get_third_party_path() / script_name
    if not path.exists():
        raise FileNotFoundError(f"Official script not found: {path}")
    return path


def get_config_path() -> Path:
    """Get path to semantic-kitti.yaml config."""
    return get_third_party_path() / "config" / "semantic-kitti.yaml"


@dataclass
class EvalResult:
    """Container for evaluation results."""

    script: str
    success: bool
    returncode: int
    stdout: str
    stderr: str
    metrics: dict[str, Any]


def parse_semantic_eval_output(stdout: str) -> dict[str, Any]:
    """Parse output from evaluate_semantics.py.

    Extracts:
    - Per-class IoU values
    - Mean IoU (mIoU)
    - Accuracy metrics

    Args:
        stdout: Raw stdout from the evaluation script.

    Returns:
        Dict with parsed metrics.
    """
    metrics: dict[str, Any] = {"per_class_iou": {}, "mean_iou": None, "accuracy": None}

    # Parse per-class IoU lines like:
    # "IoU class 1 [car] = 1.000"
    iou_pattern = re.compile(r"IoU class \d+ \[([^\]]+)\] = ([\d.]+)", re.IGNORECASE)
    for match in iou_pattern.finditer(stdout):
        class_name = match.group(1)
        iou_value = float(match.group(2))
        metrics["per_class_iou"][class_name] = iou_value

    # Parse mean IoU line like:
    # "IoU avg 1.000" or "mIoU: 0.456" or "Mean IoU: 0.456"
    miou_patterns = [
        re.compile(r"IoU avg\s+([\d.]+)", re.IGNORECASE),
        re.compile(r"mIoU\s*[:=]\s*([\d.]+)", re.IGNORECASE),
        re.compile(r"Mean\s+IoU\s*[:=]\s*([\d.]+)", re.IGNORECASE),
        re.compile(r"MEAN\s*[:=]\s*([\d.]+)", re.IGNORECASE),
    ]
    for pattern in miou_patterns:
        match = pattern.search(stdout)
        if match:
            metrics["mean_iou"] = float(match.group(1))
            break

    # Parse accuracy line like:
    # "Acc avg 1.000" or "accuracy: 0.95"
    acc_patterns = [
        re.compile(r"Acc avg\s+([\d.]+)", re.IGNORECASE),
        re.compile(r"(?:acc|accuracy)\s*[:=]\s*([\d.]+)", re.IGNORECASE),
    ]
    for pattern in acc_patterns:
        match = pattern.search(stdout)
        if match:
            metrics["accuracy"] = float(match.group(1))
            break

    return metrics


def parse_distance_eval_output(
    stdout: str, class_names: list[str] | None = None
) -> dict[str, Any]:
    """Parse output from evaluate_semantics_by_distance.py.

    The official script outputs CSV-style lines like:
        range 1e-08m to 10.0m,0.950,0.880,...,0.856,0.923

    Each line contains:
    - Distance range header
    - Per-class IoU values (depends on datacfg subset)
    - Mean IoU
    - Accuracy

    Class order follows learning_map_inv in the datacfg. If class_names is
    provided, it is used directly; otherwise we fall back to the standard
    SemanticKITTI 19-class order when the count matches.

    Args:
        stdout: Raw stdout from the evaluation script.
        class_names: Optional ordered class names from datacfg.

    Returns:
        Dict with parsed distance-stratified metrics.
    """
    metrics: dict[str, Any] = {"distance_bins": {}}

    # Standard 19-class order (full SemanticKITTI datacfg).
    class_names_full = [
        "car",
        "bicycle",
        "motorcycle",
        "truck",
        "other-vehicle",
        "person",
        "bicyclist",
        "motorcyclist",
        "road",
        "parking",
        "sidewalk",
        "other-ground",
        "building",
        "fence",
        "vegetation",
        "trunk",
        "terrain",
        "pole",
        "traffic-sign",
    ]

    # Distance bins: map from script output format to our normalized format
    # Script uses: "1e-08m to 10.0m", "10.0m to 20.0m", etc.
    distance_mapping = {
        (0, 10): "0-10m",
        (10, 20): "10-20m",
        (20, 30): "20-30m",
        (30, 40): "30-40m",
        (40, 50): "40-50m",
    }

    # Initialize all bins
    for bin_name in distance_mapping.values():
        metrics["distance_bins"][bin_name] = {
            "per_class_iou": {},
            "mean_iou": None,
            "accuracy": None,
        }

    # Parse each line
    # Pattern: "range {lrange}m to {hrange}m,{iou1},{iou2},...,{mean_iou},{accuracy}"
    range_pattern = re.compile(
        r"range\s+([\d.e+-]+)m\s+to\s+([\d.e+-]+)m,(.+)", re.IGNORECASE
    )

    for line in stdout.split("\n"):
        match = range_pattern.match(line.strip())
        if not match:
            continue

        try:
            lrange = float(match.group(1))
            hrange = float(match.group(2))
            values_str = match.group(3)

            # Parse CSV values
            values = [float(v.strip()) for v in values_str.split(",") if v.strip()]

            # Determine class names based on value count
            # values = n_classes + mean_iou + accuracy
            n_values = len(values)
            if class_names is not None and n_values == len(class_names) + 2:
                resolved_names = class_names
            elif n_values == len(class_names_full) + 2:
                resolved_names = class_names_full
            else:
                # Try to infer: last 2 are mean_iou and accuracy
                n_classes = n_values - 2
                resolved_names = [f"class_{i + 1}" for i in range(n_classes)]

            if n_values < 3:  # Need at least 1 class + mean + acc
                continue

            # Map to our normalized bin name
            bin_key = (int(round(lrange)), int(round(hrange)))
            # Handle 1e-08 as 0
            if lrange < 1:
                bin_key = (0, int(round(hrange)))

            bin_name = distance_mapping.get(bin_key)
            if not bin_name:
                continue

            # Extract per-class IoU
            for i, class_name in enumerate(resolved_names):
                metrics["distance_bins"][bin_name]["per_class_iou"][class_name] = (
                    values[i]
                )

            # Last two values are mean_iou and accuracy
            metrics["distance_bins"][bin_name]["mean_iou"] = values[len(resolved_names)]
            metrics["distance_bins"][bin_name]["accuracy"] = values[
                len(resolved_names) + 1
            ]

        except (ValueError, IndexError):
            # Skip malformed lines
            continue

    return metrics


def _class_names_from_datacfg(datacfg: Path | None) -> list[str] | None:
    """Load ordered class names from a SemanticKITTI datacfg file."""
    if datacfg is None:
        return None

    try:
        import yaml

        with open(datacfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return None

    labels_raw = cfg.get("labels", {}) or {}
    map_inv_raw = cfg.get("learning_map_inv", {}) or {}
    ignore_raw = cfg.get("learning_ignore", {}) or {}

    labels = {int(k): str(v) for k, v in labels_raw.items()}
    map_inv = {int(k): int(v) for k, v in map_inv_raw.items()}
    ignore = {int(k): bool(v) for k, v in ignore_raw.items()}

    class_names: list[str] = []
    for train_id in sorted(map_inv.keys()):
        if ignore.get(train_id, False):
            continue
        sem_id = map_inv[train_id]
        class_names.append(labels.get(sem_id, f"class_{train_id}"))

    return class_names or None


def run_semantic_eval(
    dataset: Path,
    predictions: Path,
    split: str,
    backend: str = "numpy",
    datacfg: Path | None = None,
) -> EvalResult:
    """Run evaluate_semantics.py.

    Args:
        dataset: Path to KITTI dataset root.
        predictions: Path to predictions root.
        split: Evaluation split (train/valid/test).
        backend: Evaluation backend (numpy/torch).
        datacfg: Path to custom data config file (optional).

    Returns:
        EvalResult with parsed metrics.
    """
    script = get_script_path("evaluate_semantics.py")
    config = Path(datacfg).resolve() if datacfg else get_config_path()

    # Convert paths to absolute to avoid issues with cwd change
    dataset_abs = Path(dataset).resolve()
    predictions_abs = Path(predictions).resolve()

    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        str(dataset_abs),
        "--predictions",
        str(predictions_abs),
        "--split",
        split,
        "--backend",
        backend,
        "--datacfg",
        str(config),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(script.parent),
    )

    metrics = parse_semantic_eval_output(result.stdout)

    return EvalResult(
        script="evaluate_semantics.py",
        success=result.returncode == 0,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        metrics=metrics,
    )


def run_distance_eval(
    dataset: Path,
    predictions: Path,
    split: str,
    backend: str = "numpy",
    datacfg: Path | None = None,
) -> EvalResult:
    """Run evaluate_semantics_by_distance.py.

    Args:
        dataset: Path to KITTI dataset root.
        predictions: Path to predictions root.
        split: Evaluation split (train/valid/test).
        backend: Evaluation backend (numpy/torch).
        datacfg: Path to custom data config file (optional).

    Returns:
        EvalResult with parsed distance-stratified metrics.
    """
    script = get_script_path("evaluate_semantics_by_distance.py")
    config = Path(datacfg).resolve() if datacfg else get_config_path()

    # Convert paths to absolute to avoid issues with cwd change
    dataset_abs = Path(dataset).resolve()
    predictions_abs = Path(predictions).resolve()

    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        str(dataset_abs),
        "--predictions",
        str(predictions_abs),
        "--split",
        split,
        "--backend",
        backend,
        "--datacfg",
        str(config),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(script.parent),
    )

    class_names = _class_names_from_datacfg(config)
    metrics = parse_distance_eval_output(result.stdout, class_names)

    return EvalResult(
        script="evaluate_semantics_by_distance.py",
        success=result.returncode == 0,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        metrics=metrics,
    )


def run_full_evaluation(
    dataset: Path,
    predictions: Path,
    split: str = "valid",
    output_path: Path | None = None,
    backend: str = "numpy",
    skip_semantic: bool = False,
    skip_distance: bool = False,
    datacfg: Path | None = None,
) -> dict[str, Any]:
    """Run full official SemanticKITTI evaluation pipeline.

    Args:
        dataset: Path to KITTI dataset root.
        predictions: Path to predictions root.
        split: Evaluation split (train/valid/test).
        output_path: Path to save JSON results (optional).
        backend: Evaluation backend (numpy/torch).
        skip_semantic: Skip overall semantic evaluation.
        skip_distance: Skip distance-stratified evaluation.
        datacfg: Path to custom data config file (optional).

    Returns:
        Dict with all evaluation results.
    """
    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(dataset),
        "predictions": str(predictions),
        "split": split,
        "backend": backend,
        "datacfg": str(datacfg) if datacfg else "default",
    }

    # Run semantic evaluation
    if not skip_semantic:
        print("\n" + "=" * 60)
        print("Running Semantic Evaluation")
        print("=" * 60)
        semantic_result = run_semantic_eval(
            dataset, predictions, split, backend, datacfg
        )

        results["semantic_eval"] = {
            "success": semantic_result.success,
            "returncode": semantic_result.returncode,
            "metrics": semantic_result.metrics,
        }

        if semantic_result.success:
            print("\nSemantic Evaluation Results:")
            if semantic_result.metrics.get("mean_iou") is not None:
                print(f"  Mean IoU: {semantic_result.metrics['mean_iou']:.4f}")
            if semantic_result.metrics.get("accuracy") is not None:
                print(f"  Accuracy: {semantic_result.metrics['accuracy']:.4f}")

        else:
            print(f"\nSemantic evaluation failed (code {semantic_result.returncode})")
            print(f"stderr: {semantic_result.stderr}")

    # Run distance-stratified evaluation
    if not skip_distance:
        print("\n" + "=" * 60)
        print("Running Distance-Stratified Evaluation")
        print("=" * 60)
        distance_result = run_distance_eval(
            dataset, predictions, split, backend, datacfg
        )

        results["distance_eval"] = {
            "success": distance_result.success,
            "returncode": distance_result.returncode,
            "metrics": distance_result.metrics,
        }

        if distance_result.success:
            print("\nDistance-Stratified Results:")
            for dist_range, metrics in distance_result.metrics.get(
                "distance_bins", {}
            ).items():
                miou = metrics.get("mean_iou")
                if miou is not None:
                    print(f"  {dist_range}: mIoU = {miou:.4f}")

        else:
            print(f"\nDistance evaluation failed (code {distance_result.returncode})")
            print(f"stderr: {distance_result.stderr}")

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run official SemanticKITTI evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation
  python tools/run_official_eval.py --dataset /path/to/kitti --predictions outputs/predictions

  # Run only distance-stratified evaluation
  python tools/run_official_eval.py --dataset /path/to/kitti --predictions outputs/predictions --distance-only

  # Specify output path
  python tools/run_official_eval.py --dataset /path/to/kitti --predictions outputs/predictions --output results.json
""",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        required=True,
        help="Path to KITTI dataset root (containing sequences/)",
    )
    parser.add_argument(
        "--predictions",
        "-p",
        type=Path,
        required=True,
        help="Path to predictions root (containing sequences/<seq>/predictions/)",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="valid",
        choices=["train", "valid", "test"],
        help="Evaluation split (default: valid)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs/reports/official_metrics.json"),
        help="Output JSON file path (default: outputs/reports/official_metrics.json)",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="numpy",
        choices=["numpy", "torch"],
        help="Evaluation backend (default: numpy)",
    )
    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help="Run only semantic evaluation (skip distance-stratified)",
    )
    parser.add_argument(
        "--distance-only",
        action="store_true",
        help="Run only distance-stratified evaluation (skip semantic)",
    )
    parser.add_argument(
        "--datacfg",
        type=Path,
        default=None,
        help="Path to custom data config file (default: semantic-kitti.yaml)",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.dataset.exists():
        print(f"Error: Dataset path does not exist: {args.dataset}")
        return 1
    if not args.predictions.exists():
        print(f"Error: Predictions path does not exist: {args.predictions}")
        return 1
    if args.datacfg and not args.datacfg.exists():
        print(f"Error: Data config file does not exist: {args.datacfg}")
        return 1

    try:
        results = run_full_evaluation(
            dataset=args.dataset,
            predictions=args.predictions,
            split=args.split,
            output_path=args.output,
            backend=args.backend,
            skip_semantic=args.distance_only,
            skip_distance=args.semantic_only,
            datacfg=args.datacfg,
        )

        # Determine overall success
        success = True
        if "semantic_eval" in results and not results["semantic_eval"]["success"]:
            success = False
        if "distance_eval" in results and not results["distance_eval"]["success"]:
            success = False

        return 0 if success else 1

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
