"""Run the configured detection, evaluation, and export workflow.

Steps fail fast so evaluation never consumes stale outputs after a failed run.
The optional ablation uses its own explicitly supplied configuration.

Usage:
    uv run python tools/run_all.py
    uv run python tools/run_all.py --config examples/kitti08_subset.yaml
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
    print(f"\n{'=' * 60}", flush=True)
    print(f"Step: {description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    print(flush=True)

    result = subprocess.run(cmd, cwd=REPO_ROOT)
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
        "--skip-export",
        "--skip-video",
        dest="skip_export",
        action="store_true",
        help="Skip the configured export step",
    )
    parser.add_argument(
        "--ablation-config",
        type=Path,
        default=None,
        help="Run ablation with this explicit config after the main workflow",
    )

    args = parser.parse_args()

    config_path = args.config.resolve() if args.config is not None else None
    ablation_config = (
        args.ablation_config.resolve() if args.ablation_config is not None else None
    )
    cfg = load_config(config_path=config_path, task="tools")
    dataset_root = Path(cfg["dataset"]["root"])
    if not dataset_root.is_absolute():
        dataset_root = REPO_ROOT / dataset_root
    sequence = str(cfg["dataset"]["sequence"])
    output_root = Path(cfg["output"]["root"])
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

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
    print(f"Config: {config_path or '(configs/*.yaml)'}")

    python = sys.executable
    steps_completed = 0

    # Step 1: Check dataset
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
    if not success:
        print("\nDataset validation failed; stopping before detection.")
        return 1
    steps_completed += 1

    # Step 2: Run detection pipeline
    if not args.skip_detection:
        cmd = [python, "-m", "src.main", "run"]
        if config_path is not None:
            cmd.extend(["--config", str(config_path)])
        success = run_command(cmd, "Detection Pipeline")
        if not success:
            print("\nDetection failed; stopping before evaluation.")
            return 1
        steps_completed += 1
    else:
        print(
            "\n--- Skipping detection; evaluation will use explicitly "
            "pre-existing boxes ---"
        )

    # Step 3: Run evaluation
    if not args.skip_eval:
        cmd = [python, "-m", "src.main", "eval"]
        if config_path is not None:
            cmd.extend(["--config", str(config_path)])
        success = run_command(cmd, "Evaluation")
        if not success:
            print("\nEvaluation failed; stopping before export.")
            return 1
        steps_completed += 1
    else:
        print("\n--- Skipping evaluation ---")

    # Step 4: Export videos
    if not args.skip_export:
        cmd = [python, "-m", "src.main", "export"]
        if config_path is not None:
            cmd.extend(["--config", str(config_path)])
        success = run_command(cmd, "Configured Export")
        if not success:
            print("\nExport failed; stopping.")
            return 1
        steps_completed += 1
    else:
        print("\n--- Skipping configured export ---")

    # Step 5: Run an explicitly configured ablation study.
    if ablation_config is not None:
        success = run_command(
            [
                python,
                "tools/run_ablation.py",
                "--config",
                str(ablation_config),
            ],
            "Ablation Study",
        )
        if not success:
            print("\nAblation failed.")
            return 1
        steps_completed += 1
    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Steps completed: {steps_completed}")
    print()
    print("Generated outputs:")
    if not args.skip_detection:
        boxes_dir = cfg["output"].get("boxes_dir", "boxes")
        print(f"  - {output_root / boxes_dir / sequence}/     Detection boxes")
    if not args.skip_eval:
        reports_dir = cfg["output"].get("reports_dir", "reports")
        print(f"  - {output_root / reports_dir}/     Evaluation reports")
    if not args.skip_export:
        print("  - See the configured export path for generated export artifacts")

    return 0


if __name__ == "__main__":
    sys.exit(main())
