"""Run all pipeline steps: detection, evaluation, and video export.

This script provides a one-command way to reproduce all results.

Usage:
    uv run python tools/run_all.py
    uv run python tools/run_all.py --config configs/config.local.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'=' * 60}")
    print(f"Step: {description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    success = result.returncode == 0

    if success:
        print(f"✓ {description} completed successfully")
    else:
        print(f"✗ {description} failed with code {result.returncode}")

    return success


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run complete pipeline: detection, evaluation, video export"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a single YAML config file (default: use configs/*.yaml)",
    )
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Skip detection pipeline (use existing predictions)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip video export",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip ablation study",
    )

    args = parser.parse_args()

    cfg = load_config(config_path=args.config, task="tools")
    dataset_root = Path(cfg["dataset"]["root"])
    sequence = str(cfg["dataset"]["sequence"])
    output_root = Path(cfg["output"]["root"])

    if not dataset_root.exists():
        print(f"\nError: Dataset root not found at '{dataset_root}'")
        print("Tip: Create a symbolic link named 'data' in the project root:")
        print(
            '  Windows: New-Item -ItemType Junction -Path "data" -Target "D:\\Path\\To\\Kitti"'
        )
        print("  Linux:   ln -s /path/to/kitti data")
        print("Or create 'configs/config.local.yaml' to override dataset.root.\n")
        sys.exit(1)

    print("=" * 60)
    print("pcd-detect: Full Pipeline Execution")
    print("=" * 60)
    print(f"Dataset: {dataset_root}")
    print(f"Sequence: {sequence}")
    print(f"Config: {args.config or '(configs/*.yaml)'}")

    python = sys.executable
    steps_passed = 0
    steps_total = 0

    # Step 1: Check dataset
    steps_total += 1
    success = run_command(
        [
            python,
            "tools/check_dataset.py",
            "--dataset",
            str(dataset_root),
            "--sequence",
            sequence,
        ],
        "Dataset Validation",
    )
    if success:
        steps_passed += 1
    else:
        print("\n⚠ Dataset validation failed. Continuing anyway...")

    # Step 2: Run detection pipeline
    if not args.skip_detection:
        steps_total += 1
        cmd = [python, "-m", "src.main", "run"]
        if args.config is not None:
            cmd.extend(["--config", str(args.config)])
        success = run_command(cmd, "Detection Pipeline")
        if success:
            steps_passed += 1
    else:
        print("\n--- Skipping detection pipeline ---")

    # Step 3: Run evaluation
    if not args.skip_eval:
        steps_total += 1
        cmd = [python, "-m", "src.main", "eval"]
        if args.config is not None:
            cmd.extend(["--config", str(args.config)])
        success = run_command(cmd, "Evaluation")
        if success:
            steps_passed += 1
    else:
        print("\n--- Skipping evaluation ---")

    # Step 4: Export videos
    if not args.skip_video:
        steps_total += 1
        cmd = [python, "-m", "src.main", "export"]
        if args.config is not None:
            cmd.extend(["--config", str(args.config)])
        success = run_command(cmd, "Video Export")
        if success:
            steps_passed += 1
    else:
        print("\n--- Skipping video export ---")

    # Step 5: Run ablation study
    if not args.skip_ablation:
        steps_total += 1
        success = run_command(
            [
                python,
                "tools/run_ablation.py",
                "--config",
                "configs/ablation/proposal_cmp.yaml",
            ],
            "Ablation Study",
        )
        if success:
            steps_passed += 1
    else:
        print("\n--- Skipping ablation study ---")

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Steps passed: {steps_passed}/{steps_total}")
    print()
    print("Generated outputs:")
    print(f"  - {output_root / cfg['output']['predictions_dir']}/     Predictions")
    print(f"  - {output_root / cfg['output']['reports_dir']}/         Reports")
    print(f"  - {output_root / cfg['output']['videos_dir']}/          Videos")
    print()
    print("Quick visual check:")
    print("  python third_party/semkitti_api/visualize.py \\")
    print(f"      --sequence {sequence} --dataset {dataset_root} \\")
    print(f"      --predictions {output_root / cfg['output']['predictions_dir']}")

    return 0 if steps_passed == steps_total else 1


if __name__ == "__main__":
    sys.exit(main())
