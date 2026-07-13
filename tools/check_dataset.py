#!/usr/bin/env python3
"""
Dataset consistency checker for KITTI Odometry + SemanticKITTI.

This script validates the dataset structure and content integrity:
- Directory structure completeness
- Point cloud file format (.bin as N×4 float32)
- Label file format (.label as N uint32)
- Point count consistency between .bin and .label files
- Semantic ID distribution sanity check
- Instance ID temporal continuity check

Usage:
    uv run python tools/check_dataset.py --dataset /path/to/dataset --sequence 00
    uv run python tools/check_dataset.py --dataset /path/to/dataset --all-sequences
    uv run python tools/check_dataset.py --dataset /path/to/dataset --sequence 00 --sample 10
"""

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger  # noqa: E402
from src.datasets.semkitti_labels import SemanticKITTILabels  # noqa: E402

logger = get_logger(__name__)


class CheckResult(NamedTuple):
    """Result of a single check."""

    passed: bool
    message: str
    details: dict | None = None


class DatasetChecker:
    """
    Validates KITTI Odometry + SemanticKITTI dataset structure and content.

    Performs the following checks:
    1. Directory structure: velodyne/, labels/ (optional), calib.txt, poses.txt
    2. File counts: number of .bin and .label files match
    3. File format: .bin is N×4 float32, .label is N uint32
    4. Point-label consistency: number of points matches number of labels
    5. Semantic ID range: all semantic IDs are within expected range
    6. Instance continuity: instances persist across frames (sampling check)
    """

    # Fallback IDs if config cannot be loaded.
    VALID_SEMANTIC_IDS = {
        0,
        1,
        10,
        11,
        13,
        15,
        16,
        18,
        20,
        30,
        31,
        32,
        40,
        44,
        48,
        49,
        50,
        51,
        52,
        60,
        70,
        71,
        72,
        80,
        81,
        99,
        252,
        253,
        254,
        255,
        256,
        257,
        258,
        259,
    }

    def __init__(self, dataset_root: str | Path, require_poses: bool = False):
        """
        Initialize dataset checker.

        Args:
            dataset_root: Path to dataset root (containing 'sequences' folder).
            require_poses: If True, require `poses.txt` for sequences 00-10.
        """
        self.root = Path(dataset_root)
        self.results: list[CheckResult] = []
        self.require_poses = require_poses
        self._valid_semantic_ids = self._load_valid_semantic_ids()
        self._thing_semantic_ids = self._load_thing_semantic_ids()

    def _load_valid_semantic_ids(self) -> set[int]:
        """Load valid semantic ids from official SemanticKITTI config."""
        try:
            labels = SemanticKITTILabels()
            return set(int(k) for k in labels.labels.keys())
        except Exception:
            return set(self.VALID_SEMANTIC_IDS)

    def _load_thing_semantic_ids(self) -> set[int]:
        """Load 'thing' semantic ids (object classes) used for instance continuity."""
        try:
            return set(int(k) for k in SemanticKITTILabels.THING_CLASSES.keys())
        except Exception:
            return {
                10,
                11,
                13,
                15,
                18,
                20,
                30,
                31,
                32,
                252,
                253,
                254,
                255,
                257,
                258,
                259,
            }

    def check_sequence(
        self,
        sequence: str,
        sample_size: int | None = None,
        verbose: bool = True,
    ) -> tuple[bool, list[CheckResult]]:
        """
        Run all checks on a single sequence.

        Args:
            sequence: Sequence ID (e.g., "00").
            sample_size: Number of frames to sample. If None, check all frames.
            verbose: If True, print progress messages.

        Returns:
            Tuple of (all_passed, list of CheckResults).
        """
        self.results = []
        sequence = sequence.zfill(2)
        seq_path = self.root / "sequences" / sequence

        if verbose:
            logger.info(f"Checking sequence {sequence} at {seq_path}")

        # Check 1: Directory structure
        self._check_directory_structure(seq_path)

        velodyne_path = seq_path / "velodyne"
        label_path = seq_path / "labels"

        # Check 2: File counts
        bin_files = (
            sorted(velodyne_path.glob("*.bin")) if velodyne_path.exists() else []
        )
        label_files = sorted(label_path.glob("*.label")) if label_path.exists() else []

        self._check_file_counts(bin_files, label_files)

        # If no files, stop here
        if not bin_files:
            all_passed = all(r.passed for r in self.results)
            return all_passed, self.results

        # Determine frames to check
        if sample_size is not None and sample_size < len(bin_files):
            # Sample evenly across the sequence
            indices = np.linspace(0, len(bin_files) - 1, sample_size, dtype=int)
        else:
            indices = list(range(len(bin_files)))

        if verbose:
            logger.info(f"Checking {len(indices)} frames out of {len(bin_files)}")

        # Check each sampled frame
        for idx in indices:
            frame_id = int(bin_files[idx].stem)
            self._check_frame(velodyne_path, label_path, frame_id, verbose)

        # Check 3: Instance continuity (if labels exist)
        if label_files:
            self._check_instance_continuity(
                label_path, sample_size=min(10, len(label_files))
            )

        all_passed = all(r.passed for r in self.results)
        return all_passed, self.results

    def _check_directory_structure(self, seq_path: Path) -> None:
        """Check that required directories and files exist."""
        # Sequence directory
        if not seq_path.exists():
            self.results.append(
                CheckResult(
                    passed=False,
                    message=f"Sequence directory not found: {seq_path}",
                )
            )
            return

        # Velodyne directory (required)
        velodyne_path = seq_path / "velodyne"
        if not velodyne_path.exists():
            self.results.append(
                CheckResult(
                    passed=False,
                    message=f"Velodyne directory not found: {velodyne_path}",
                )
            )
        else:
            self.results.append(
                CheckResult(
                    passed=True,
                    message="Velodyne directory exists",
                )
            )

        # Labels directory (optional but expected for SemanticKITTI)
        label_path = seq_path / "labels"
        if label_path.exists():
            self.results.append(
                CheckResult(
                    passed=True,
                    message="Labels directory exists",
                )
            )
        else:
            self.results.append(
                CheckResult(
                    passed=True,
                    message="Labels directory not found (optional)",
                )
            )

        # Calibration file (optional)
        calib_path = seq_path / "calib.txt"
        if calib_path.exists():
            self.results.append(
                CheckResult(
                    passed=True,
                    message="Calibration file exists",
                )
            )
        else:
            self.results.append(
                CheckResult(
                    passed=True,
                    message="Calibration file not found (optional for some sequences)",
                )
            )

        # Poses file (optional, only sequences 00-10 have ground truth)
        poses_path = seq_path / "poses.txt"
        if poses_path.exists():
            self.results.append(
                CheckResult(
                    passed=True,
                    message="Poses file exists",
                )
            )
        else:
            seq_num = int(seq_path.name)
            if seq_num <= 10 and self.require_poses:
                self.results.append(
                    CheckResult(
                        passed=False,
                        message=f"Poses file not found for sequence {seq_num} (expected)",
                    )
                )
            else:
                self.results.append(
                    CheckResult(
                        passed=True,
                        message="Poses file not found (optional)",
                    )
                )

    def _check_file_counts(
        self, bin_files: list[Path], label_files: list[Path]
    ) -> None:
        """Check that .bin and .label file counts match."""
        n_bins = len(bin_files)
        n_labels = len(label_files)

        if n_bins == 0:
            self.results.append(
                CheckResult(
                    passed=False,
                    message="No .bin files found",
                )
            )
            return

        self.results.append(
            CheckResult(
                passed=True,
                message=f"Found {n_bins} point cloud files",
            )
        )

        if n_labels == 0:
            self.results.append(
                CheckResult(
                    passed=True,
                    message="No .label files found (unlabeled sequence)",
                )
            )
        elif n_bins == n_labels:
            self.results.append(
                CheckResult(
                    passed=True,
                    message=f"Point cloud and label counts match: {n_bins}",
                )
            )
        else:
            self.results.append(
                CheckResult(
                    passed=False,
                    message=f"Mismatch: {n_bins} .bin files vs {n_labels} .label files",
                )
            )

    def _check_frame(
        self,
        velodyne_path: Path,
        label_path: Path,
        frame_id: int,
        verbose: bool,
    ) -> None:
        """Check a single frame's point cloud and label files."""
        bin_path = velodyne_path / f"{frame_id:06d}.bin"
        label_file = label_path / f"{frame_id:06d}.label"

        # Check .bin file
        if not bin_path.exists():
            self.results.append(
                CheckResult(
                    passed=False,
                    message=f"Frame {frame_id}: .bin file missing",
                )
            )
            return

        # Read and validate .bin
        try:
            file_size = bin_path.stat().st_size
            if file_size % 16 != 0:  # 4 float32 values × 4 bytes
                self.results.append(
                    CheckResult(
                        passed=False,
                        message=f"Frame {frame_id}: .bin size {file_size} not divisible by 16",
                    )
                )
                return

            n_points = file_size // 16
            points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)

            if points.shape[0] != n_points:
                self.results.append(
                    CheckResult(
                        passed=False,
                        message=f"Frame {frame_id}: point count mismatch",
                    )
                )
                return

            # Check for NaN/Inf values
            if np.any(~np.isfinite(points)):
                self.results.append(
                    CheckResult(
                        passed=False,
                        message=f"Frame {frame_id}: contains NaN/Inf values",
                    )
                )
                return

        except Exception as e:
            self.results.append(
                CheckResult(
                    passed=False,
                    message=f"Frame {frame_id}: failed to read .bin - {e}",
                )
            )
            return

        # Check .label file if exists
        if label_file.exists():
            try:
                label_size = label_file.stat().st_size
                if label_size % 4 != 0:  # 1 uint32 per point
                    self.results.append(
                        CheckResult(
                            passed=False,
                            message=f"Frame {frame_id}: .label size {label_size} not divisible by 4",
                        )
                    )
                    return

                n_labels = label_size // 4
                labels = np.fromfile(str(label_file), dtype=np.uint32)

                # Check point-label consistency
                if n_points != n_labels:
                    self.results.append(
                        CheckResult(
                            passed=False,
                            message=f"Frame {frame_id}: {n_points} points vs {n_labels} labels",
                        )
                    )
                    return

                # Check semantic ID range
                sem_labels = labels & 0xFFFF
                unique_sem = set(np.unique(sem_labels))
                invalid_ids = unique_sem - self._valid_semantic_ids
                if invalid_ids:
                    self.results.append(
                        CheckResult(
                            passed=False,
                            message=f"Frame {frame_id}: invalid semantic IDs: {invalid_ids}",
                        )
                    )
                    return

            except Exception as e:
                self.results.append(
                    CheckResult(
                        passed=False,
                        message=f"Frame {frame_id}: failed to read .label - {e}",
                    )
                )
                return

        if verbose and frame_id % 100 == 0:
            logger.debug(f"Frame {frame_id}: OK ({n_points} points)")

    def _check_instance_continuity(
        self, label_path: Path, sample_size: int = 10
    ) -> None:
        """
        Check that instance IDs persist across consecutive frames.

        For moving objects, the same instance ID should appear in multiple
        consecutive frames, allowing for tracking.
        """
        label_files = sorted(label_path.glob("*.label"))
        if len(label_files) < 2:
            return

        # Sample consecutive frame pairs
        n_pairs = min(sample_size, len(label_files) - 1)
        pair_indices = np.linspace(0, len(label_files) - 2, n_pairs, dtype=int)

        continuous_instances = 0
        total_instances = 0

        for idx in pair_indices:
            try:
                labels1 = np.fromfile(str(label_files[idx]), dtype=np.uint32)
                labels2 = np.fromfile(str(label_files[idx + 1]), dtype=np.uint32)

                # Extract instance IDs (only for "thing" classes)
                sem1 = labels1 & 0xFFFF
                sem2 = labels2 & 0xFFFF
                inst1 = labels1 >> 16
                inst2 = labels2 >> 16

                # Get unique instance IDs for thing classes
                thing_ids = np.array(sorted(self._thing_semantic_ids), dtype=np.uint32)
                thing_mask1 = np.isin(sem1, thing_ids)
                thing_mask2 = np.isin(sem2, thing_ids)

                unique_inst1 = set(inst1[thing_mask1]) - {0}
                unique_inst2 = set(inst2[thing_mask2]) - {0}

                # Check overlap
                overlap = unique_inst1 & unique_inst2
                continuous_instances += len(overlap)
                total_instances += len(unique_inst1)

            except Exception:
                continue

        if total_instances > 0:
            continuity_rate = continuous_instances / total_instances
            if continuity_rate < 0.5:  # Less than 50% continuity is suspicious
                self.results.append(
                    CheckResult(
                        passed=False,
                        message=f"Low instance continuity: {continuity_rate:.1%}",
                        details={"continuity_rate": continuity_rate},
                    )
                )
            else:
                self.results.append(
                    CheckResult(
                        passed=True,
                        message=f"Instance continuity check passed: {continuity_rate:.1%}",
                    )
                )
        else:
            self.results.append(
                CheckResult(
                    passed=True,
                    message="No thing instances found for continuity check",
                )
            )


def print_results(results: list[CheckResult]) -> None:
    """Print check results in a formatted way."""
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print("\n" + "=" * 60)
    print("DATASET CHECK RESULTS")
    print("=" * 60)

    for r in results:
        status = "OK" if r.passed else "FAIL"
        color = "\033[92m" if r.passed else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {r.message}")

    print("=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Check KITTI + SemanticKITTI dataset consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Check single sequence:
    uv run python tools/check_dataset.py --dataset /path/to/dataset --sequence 00

  Check all sequences:
    uv run python tools/check_dataset.py --dataset /path/to/dataset --all-sequences

  Sample 10 frames per sequence:
    uv run python tools/check_dataset.py --dataset /path/to/dataset --sequence 00 --sample 10
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset root (containing 'sequences' folder)",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Sequence to check (e.g., '00')",
    )
    parser.add_argument(
        "--all-sequences",
        action="store_true",
        help="Check all available sequences",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of frames to sample per sequence (default: all)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--require-poses",
        action="store_true",
        help="Require poses.txt for sequences 00-10 (strict KITTI Odometry)",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        print(f"ERROR: Dataset root not found: {dataset_root}")
        sys.exit(1)

    sequences_path = dataset_root / "sequences"
    if not sequences_path.exists():
        print(f"ERROR: 'sequences' folder not found in {dataset_root}")
        sys.exit(1)

    checker = DatasetChecker(dataset_root, require_poses=args.require_poses)

    # Determine sequences to check
    if args.all_sequences:
        sequences = sorted([d.name for d in sequences_path.iterdir() if d.is_dir()])
    elif args.sequence:
        sequences = [args.sequence]
    else:
        print("ERROR: Specify --sequence or --all-sequences")
        sys.exit(1)

    all_passed = True
    total_passed = 0
    total_failed = 0

    for seq in sequences:
        passed, results = checker.check_sequence(
            seq, sample_size=args.sample, verbose=not args.quiet
        )
        print("\n" + "=" * 60)
        print(f"SEQUENCE {seq}")
        print("=" * 60)

        for r in results:
            status = "\u2713" if r.passed else "\u2717"
            color = "\033[92m" if r.passed else "\033[91m"
            reset = "\033[0m"
            print(f"{color}{status}{reset} {r.message}")

        seq_passed = sum(1 for r in results if r.passed)
        seq_failed = len(results) - seq_passed
        total_passed += seq_passed
        total_failed += seq_failed

        print(f"\nSequence {seq}: {seq_passed} passed, {seq_failed} failed")
        all_passed = all_passed and passed

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Checked {len(sequences)} sequence(s)")
    print(f"Total: {total_passed} passed, {total_failed} failed")
    print(f"Status: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
