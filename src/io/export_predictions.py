"""src.io.export_predictions

Export helpers for SemanticKITTI-compatible predictions.

The official SemanticKITTI API expects the following structure:

  <predictions_root>/sequences/<SEQ>/predictions/<FRAME>.label

Each .label file is a uint32 array with one value per point.
For semantic segmentation evaluation, only the lower 16 bits are used
(semantic class id). We still write a full uint32 label, keeping the
upper 16 bits as instance id (default 0).

Usage:
    # Export entire sequence
    exporter = PredictionExporter(
        predictions_root="outputs/predictions",
        sequence="00",
        dataset_root="/path/to/kitti/dataset"
    )
    exporter.export_all_frames()

    # Validate structure
    validator = PredictionValidator(predictions_root, dataset_root)
    is_valid, errors = validator.validate_sequence("00")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from numpy.typing import NDArray

from ..datasets.kitti_odometry import KITTIDataset
from ..datasets.label_maps import SemanticKITTIMaps
from ..datasets.semkitti_labels import write_label_file
from ..utils.logging import get_logger

logger = get_logger(__name__)


def prediction_label_path(
    predictions_root: str | Path, sequence: str, frame_id: int
) -> Path:
    """Build path to prediction label file.

    Args:
        predictions_root: Root directory for predictions.
        sequence: Sequence ID (e.g., "00").
        frame_id: Frame index.

    Returns:
        Path like: predictions_root/sequences/00/predictions/000000.label
    """
    root = Path(predictions_root)
    seq = str(sequence).zfill(2)
    return root / "sequences" / seq / "predictions" / f"{frame_id:06d}.label"


def export_frame_predictions(
    *,
    predictions_root: str | Path,
    sequence: str,
    frame_id: int,
    semantic_ids: NDArray[np.integer],
    instance_ids: NDArray[np.integer] | None = None,
    semantic_ids_are_training: bool = False,
    maps: SemanticKITTIMaps | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """Export a single frame prediction file in SemanticKITTI format.

    Args:
        predictions_root: Output root folder.
        sequence: Sequence id (e.g., "00").
        frame_id: Frame index.
        semantic_ids: (N,) semantic ids. Either original SemanticKITTI ids or training ids.
        instance_ids: Optional (N,) instance ids to store in upper 16 bits (default zeros).
        semantic_ids_are_training: If True, apply learning_map_inv to convert to original ids.
        maps: Optional pre-loaded SemanticKITTIMaps (saves repeated YAML loads).
        config_path: Optional path to semantic-kitti.yaml.

    Returns:
        Path to written .label file.
    """
    if semantic_ids.ndim != 1:
        raise ValueError(f"semantic_ids must be 1D, got shape={semantic_ids.shape}")

    if instance_ids is not None and instance_ids.shape != semantic_ids.shape:
        raise ValueError(
            f"instance_ids shape {instance_ids.shape} does not match semantic_ids {semantic_ids.shape}"
        )

    if semantic_ids_are_training:
        if maps is None:
            maps = SemanticKITTIMaps.from_config(config_path)
        semantic_ids = maps.map_from_training(semantic_ids)

    sem_u32 = semantic_ids.astype(np.uint32, copy=False)
    mask = np.uint32(0xFFFF)

    if instance_ids is None:
        inst_u32 = np.zeros_like(sem_u32)
    else:
        inst_u32 = instance_ids.astype(np.uint32, copy=False)

    # SemanticKITTI packing: lower 16 bits semantic, upper 16 bits instance.
    packed = (sem_u32 & mask) | ((inst_u32 & mask) << np.uint32(16))

    out_path = prediction_label_path(predictions_root, sequence, frame_id)
    write_label_file(out_path, packed)
    return out_path


@dataclass
class ExportStats:
    """Statistics for prediction export operation."""

    sequence: str
    num_frames: int
    num_exported: int
    num_failed: int
    total_points: int
    failed_frames: list[int]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sequence": self.sequence,
            "num_frames": self.num_frames,
            "num_exported": self.num_exported,
            "num_failed": self.num_failed,
            "total_points": self.total_points,
            "failed_frames": self.failed_frames,
        }


class PredictionExporter:
    """Batch exporter for SemanticKITTI-compatible predictions.

    Generates the full prediction directory structure:
        predictions_root/sequences/XX/predictions/XXXXXX.label

    Supports:
    - Exporting ground truth labels as predictions (for testing)
    - Exporting detection results with semantic+instance IDs
    - Automatic label remapping via learning_map_inv
    """

    def __init__(
        self,
        predictions_root: str | Path,
        sequence: str,
        dataset_root: str | Path | None = None,
        semantic_ids_are_training: bool = False,
        config_path: str | Path | None = None,
    ):
        """Initialize prediction exporter.

        Args:
            predictions_root: Root directory for predictions output.
            sequence: Sequence ID (e.g., "00").
            dataset_root: Path to KITTI dataset (for frame count validation).
            semantic_ids_are_training: If True, apply learning_map_inv on export.
            config_path: Path to semantic-kitti.yaml config.
        """
        self.predictions_root = Path(predictions_root)
        self.sequence = str(sequence).zfill(2)
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.semantic_ids_are_training = semantic_ids_are_training

        # Load label maps if needed for remapping
        self.maps: SemanticKITTIMaps | None = None
        if semantic_ids_are_training:
            self.maps = SemanticKITTIMaps.from_config(config_path)

        # Create output directory structure
        self._predictions_dir = (
            self.predictions_root / "sequences" / self.sequence / "predictions"
        )
        self._predictions_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized PredictionExporter: sequence={self.sequence}, "
            f"output={self._predictions_dir}"
        )

    @property
    def predictions_dir(self) -> Path:
        """Return path to predictions directory for this sequence."""
        return self._predictions_dir

    def export_frame(
        self,
        frame_id: int,
        semantic_ids: NDArray[np.integer],
        instance_ids: NDArray[np.integer] | None = None,
    ) -> Path:
        """Export predictions for a single frame.

        Args:
            frame_id: Frame index.
            semantic_ids: (N,) semantic class IDs.
            instance_ids: Optional (N,) instance IDs.

        Returns:
            Path to written .label file.
        """
        return export_frame_predictions(
            predictions_root=self.predictions_root,
            sequence=self.sequence,
            frame_id=frame_id,
            semantic_ids=semantic_ids,
            instance_ids=instance_ids,
            semantic_ids_are_training=self.semantic_ids_are_training,
            maps=self.maps,
        )

    def export_from_iterator(
        self,
        frame_iterator: Iterator[
            tuple[int, NDArray[np.integer], NDArray[np.integer] | None]
        ],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ExportStats:
        """Export predictions from an iterator.

        Args:
            frame_iterator: Yields (frame_id, semantic_ids, instance_ids) tuples.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            ExportStats with export results.
        """
        num_exported = 0
        num_failed = 0
        total_points = 0
        failed_frames: list[int] = []

        frames_list = list(frame_iterator)
        num_frames = len(frames_list)

        for i, (frame_id, semantic_ids, instance_ids) in enumerate(frames_list):
            try:
                self.export_frame(frame_id, semantic_ids, instance_ids)
                num_exported += 1
                total_points += len(semantic_ids)
            except Exception as e:
                logger.error(f"Failed to export frame {frame_id}: {e}")
                num_failed += 1
                failed_frames.append(frame_id)

            if progress_callback:
                progress_callback(i + 1, num_frames)

        return ExportStats(
            sequence=self.sequence,
            num_frames=num_frames,
            num_exported=num_exported,
            num_failed=num_failed,
            total_points=total_points,
            failed_frames=failed_frames,
        )

    def export_ground_truth_as_predictions(
        self,
        dataset: KITTIDataset | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ExportStats:
        """Export ground truth labels as predictions (for validation/testing).

        This is useful for:
        - Testing the evaluation pipeline with perfect predictions
        - Validating the export/import format compatibility
        - Creating baseline predictions

        Args:
            dataset: Optional KITTIDataset. If None, uses dataset_root.
            start_frame: First frame to export (inclusive).
            end_frame: Last frame to export (exclusive). None = all frames.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            ExportStats with export results.
        """
        if dataset is None:
            if self.dataset_root is None:
                raise ValueError("Either dataset or dataset_root must be provided")
            dataset = KITTIDataset(self.dataset_root, self.sequence)

        if not dataset.has_labels:
            raise ValueError(f"Dataset sequence {self.sequence} has no labels")

        if end_frame is None:
            end_frame = dataset.num_frames

        num_frames = end_frame - start_frame
        num_exported = 0
        num_failed = 0
        total_points = 0
        failed_frames: list[int] = []

        logger.info(
            f"Exporting GT as predictions: frames {start_frame} to {end_frame - 1}"
        )

        for i, frame_id in enumerate(range(start_frame, end_frame)):
            try:
                frame_data = dataset.get_frame(frame_id)
                if frame_data.sem_label is None:
                    raise ValueError(f"Frame {frame_id} has no semantic labels")

                # Export semantic labels (optionally with instance IDs)
                self.export_frame(
                    frame_id=frame_id,
                    semantic_ids=frame_data.sem_label,
                    instance_ids=frame_data.inst_label,
                )
                num_exported += 1
                total_points += len(frame_data.sem_label)

            except Exception as e:
                logger.error(f"Failed to export frame {frame_id}: {e}")
                num_failed += 1
                failed_frames.append(frame_id)

            if progress_callback:
                progress_callback(i + 1, num_frames)

        stats = ExportStats(
            sequence=self.sequence,
            num_frames=num_frames,
            num_exported=num_exported,
            num_failed=num_failed,
            total_points=total_points,
            failed_frames=failed_frames,
        )

        logger.info(
            f"Export complete: {num_exported}/{num_frames} frames, "
            f"{total_points:,} total points"
        )

        return stats


class PredictionValidator:
    """Validator for SemanticKITTI prediction directory structure.

    Checks:
    - Directory structure matches expected format
    - Frame counts match between predictions and ground truth
    - Point counts match for each frame
    - Label values are in valid range
    """

    def __init__(
        self,
        predictions_root: str | Path,
        dataset_root: str | Path,
    ):
        """Initialize validator.

        Args:
            predictions_root: Root directory containing predictions.
            dataset_root: Root directory of KITTI dataset (for ground truth).
        """
        self.predictions_root = Path(predictions_root)
        self.dataset_root = Path(dataset_root)

    def _get_prediction_files(self, sequence: str) -> list[Path]:
        """Get list of prediction label files for a sequence."""
        seq = str(sequence).zfill(2)
        pred_dir = self.predictions_root / "sequences" / seq / "predictions"
        if not pred_dir.exists():
            return []
        return sorted(pred_dir.glob("*.label"))

    def validate_sequence(
        self, sequence: str, check_point_counts: bool = True
    ) -> tuple[bool, list[str]]:
        """Validate predictions for a sequence.

        Args:
            sequence: Sequence ID (e.g., "00").
            check_point_counts: If True, also validate point counts match.

        Returns:
            (is_valid, errors): Tuple of validation result and error messages.
        """
        errors: list[str] = []
        seq = str(sequence).zfill(2)

        # Load dataset for reference
        try:
            dataset = KITTIDataset(self.dataset_root, seq)
        except Exception as e:
            errors.append(f"Failed to load dataset: {e}")
            return False, errors

        # Check prediction directory exists
        pred_dir = self.predictions_root / "sequences" / seq / "predictions"
        if not pred_dir.exists():
            errors.append(f"Prediction directory not found: {pred_dir}")
            return False, errors

        # Get prediction files
        pred_files = self._get_prediction_files(seq)
        if not pred_files:
            errors.append(f"No prediction files found in {pred_dir}")
            return False, errors

        # Check frame count
        if len(pred_files) != dataset.num_frames:
            errors.append(
                f"Frame count mismatch: {len(pred_files)} predictions vs "
                f"{dataset.num_frames} ground truth frames"
            )

        # Validate each prediction file
        for pred_file in pred_files:
            frame_id = int(pred_file.stem)

            # Check file can be read
            try:
                pred_labels = np.fromfile(str(pred_file), dtype=np.uint32)
            except Exception as e:
                errors.append(f"Failed to read {pred_file.name}: {e}")
                continue

            # Check point count matches
            if check_point_counts and frame_id < dataset.num_frames:
                try:
                    frame_data = dataset.get_frame(frame_id)
                    expected_points = len(frame_data.points)
                    if len(pred_labels) != expected_points:
                        errors.append(
                            f"Frame {frame_id}: point count mismatch "
                            f"({len(pred_labels)} vs {expected_points})"
                        )
                except Exception as e:
                    errors.append(f"Frame {frame_id}: failed to verify - {e}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_all_sequences(
        self, sequences: list[str] | None = None
    ) -> dict[str, tuple[bool, list[str]]]:
        """Validate predictions for multiple sequences.

        Args:
            sequences: List of sequence IDs. If None, auto-detect from predictions.

        Returns:
            Dict mapping sequence ID to (is_valid, errors) tuples.
        """
        if sequences is None:
            # Auto-detect sequences from predictions directory
            seq_dir = self.predictions_root / "sequences"
            if not seq_dir.exists():
                return {}
            sequences = [d.name for d in seq_dir.iterdir() if d.is_dir()]

        results = {}
        for seq in sequences:
            results[seq] = self.validate_sequence(seq)
        return results


def create_predictions_structure(
    predictions_root: str | Path,
    sequences: list[str],
) -> None:
    """Create empty predictions directory structure.

    Creates:
        predictions_root/sequences/XX/predictions/ for each sequence

    Args:
        predictions_root: Root directory for predictions.
        sequences: List of sequence IDs.
    """
    root = Path(predictions_root)
    for seq in sequences:
        seq_dir = root / "sequences" / str(seq).zfill(2) / "predictions"
        seq_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created prediction directory: {seq_dir}")
