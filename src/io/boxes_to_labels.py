"""Convert detection boxes to SemanticKITTI-compatible label predictions.

This module bridges the gap between object detection (bounding boxes) and
semantic segmentation evaluation (per-point labels). It assigns semantic
labels to points based on whether they fall inside detected bounding boxes.

Workflow:
    1. Load point cloud and corresponding detection boxes
    2. For each point, check which box (if any) contains it
    3. Assign semantic class based on box class or detection heuristics
    4. Export as .label files in SemanticKITTI format

Class Assignment Strategy:
    - Points inside valid boxes: assigned the box's semantic class
    - Points outside all boxes: assigned a default class (unlabeled or background)

Usage:
    from src.io.boxes_to_labels import BoxesToLabelsConverter

    converter = BoxesToLabelsConverter(
        dataset_root="/path/to/kitti",
        boxes_root="outputs/boxes",
        predictions_root="outputs/predictions",
        sequence="08",
    )
    converter.convert_all_frames()
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..boxes.obb import points_in_obbs
from ..datasets.kitti_odometry import KITTIDataset
from ..datasets.semkitti_labels import write_label_file
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Default semantic class IDs (SemanticKITTI original labels)
UNLABELED_ID = 0  # unlabeled/unknown
CAR_ID = 10  # car
PERSON_ID = 30  # person
BUILDING_ID = 50  # building (for static structures)


@dataclass
class BoxLabelMapping:
    """Configuration for mapping detection boxes to semantic labels.

    Attributes:
        default_box_class: Semantic class ID for points inside boxes.
        background_class: Semantic class ID for points outside all boxes.
        use_box_size_heuristics: Whether to use box dimensions to infer class.
    """

    default_box_class: int = CAR_ID
    background_class: int = UNLABELED_ID
    use_box_size_heuristics: bool = True

    # Size thresholds for heuristic classification (in meters)
    # Based on typical object dimensions
    car_volume_range: tuple[float, float] = (2.0, 80.0)
    person_volume_range: tuple[float, float] = (0.1, 2.0)
    person_height_min: float = 1.2

    def infer_class_from_box(self, box: dict) -> int:
        """Infer semantic class from box dimensions.

        Uses volume and aspect ratio heuristics to classify:
        - Explicit semantic/class id -> that class
        - Small, tall boxes → person
        - Medium boxes → car
        - Other → default

        Args:
            box: Box dict with 'dimensions' and 'volume' keys.

        Returns:
            Inferred semantic class ID.
        """
        for key in ("semantic_id", "class_id"):
            value = box.get(key)
            if value is not None:
                semantic_id = int(value)
                if semantic_id > 0:
                    return semantic_id

        if not self.use_box_size_heuristics:
            return self.default_box_class

        volume = box.get("volume", 0)
        dimensions = box.get("dimensions", [0, 0, 0])

        # Height is the 3rd dimension (length, width, height)
        height = dimensions[2] if len(dimensions) >= 3 else 0

        # Person heuristic: small volume, reasonable height
        if (
            self.person_volume_range[0] <= volume <= self.person_volume_range[1]
            and height >= self.person_height_min
        ):
            return PERSON_ID

        # Car heuristic: medium volume
        if self.car_volume_range[0] <= volume <= self.car_volume_range[1]:
            return CAR_ID

        return self.default_box_class


@dataclass
class ConversionStats:
    """Statistics from label conversion."""

    frame_id: int
    num_points: int
    num_boxes: int
    points_in_boxes: int
    points_outside_boxes: int
    class_distribution: dict[int, int]


def load_frame_boxes(boxes_file: Path, frame_id: int) -> list[dict] | None:
    """Load boxes for a specific frame from JSONL file.

    Args:
        boxes_file: Path to boxes.jsonl file.
        frame_id: Frame ID to load.

    Returns:
        List of box dicts, or None if frame not found.
    """
    if not boxes_file.exists():
        return None

    with open(boxes_file, "r", encoding="utf-8") as f:
        for line in f:
            frame_data = json.loads(line)
            if frame_data.get("frame_id") == frame_id:
                return frame_data.get("boxes", [])

    return None


def convert_boxes_to_labels(
    points: NDArray[np.floating],
    boxes: list[dict],
    mapping: BoxLabelMapping | None = None,
) -> tuple[NDArray[np.uint32], ConversionStats]:
    """Convert detection boxes to per-point semantic labels.

    Args:
        points: (N, 3+) Point cloud array.
        boxes: List of box dicts with 'center', 'extent', 'yaw', 'valid' keys.
        mapping: Label mapping configuration. Uses default if None.

    Returns:
        labels: (N,) uint32 semantic labels.
        stats: Conversion statistics.

    Notes:
        Points inside detected boxes are assigned an inferred class (car/person).
        Points outside all boxes are assigned `mapping.background_class`.
    """
    if mapping is None:
        mapping = BoxLabelMapping()

    n_points = points.shape[0]

    labels = np.full(n_points, mapping.background_class, dtype=np.uint32)

    # Find which points are in which boxes
    box_ids, in_any_box = points_in_obbs(points, boxes, valid_only=True)

    # Build class distribution tracker
    class_counts: dict[int, int] = {mapping.background_class: 0}

    # Assign labels based on box membership
    for box_idx, box in enumerate(boxes):
        if not box.get("valid", True):
            continue

        # Get points in this box
        in_this_box = box_ids == box_idx
        if not np.any(in_this_box):
            continue

        # Infer class for this box
        semantic_class = mapping.infer_class_from_box(box)
        labels[in_this_box] = semantic_class

        # Track class counts
        count = int(np.sum(in_this_box))
        class_counts[semantic_class] = class_counts.get(semantic_class, 0) + count

    # Count unlabeled points (points outside all boxes)
    class_counts[mapping.background_class] = int(np.sum(~in_any_box))

    stats = ConversionStats(
        frame_id=-1,  # Set by caller
        num_points=n_points,
        num_boxes=len([b for b in boxes if b.get("valid", True)]),
        points_in_boxes=int(np.sum(in_any_box)),
        points_outside_boxes=int(np.sum(~in_any_box)),
        class_distribution=class_counts,
    )

    return labels, stats


class BoxesToLabelsConverter:
    """Convert detection boxes to SemanticKITTI label format.

    This converter bridges detection results (3D bounding boxes) with
    the semantic segmentation evaluation pipeline by generating per-point
    label predictions.

    Attributes:
        dataset: KITTIDataset instance.
        boxes_root: Root directory containing box results.
        predictions_root: Output directory for label predictions.
        sequence: Sequence ID.
        mapping: Label mapping configuration.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        boxes_root: str | Path,
        predictions_root: str | Path,
        sequence: str,
        mapping: BoxLabelMapping | None = None,
    ):
        """Initialize the converter.

        Args:
            dataset_root: Path to KITTI dataset root.
            boxes_root: Root directory containing box JSONL files.
            predictions_root: Output directory for predictions.
            sequence: Sequence ID (e.g., "08").
            mapping: Label mapping configuration.
        """
        self.dataset = KITTIDataset(dataset_root, sequence)
        self.boxes_root = Path(boxes_root)
        self.predictions_root = Path(predictions_root)
        self.sequence = str(sequence).zfill(2)
        self.mapping = mapping or BoxLabelMapping()
        self._boxes_by_frame: dict[int, list[dict]] | None = None

        # Build paths
        self.boxes_file = self.boxes_root / self.sequence / "boxes.jsonl"
        self.output_dir = (
            self.predictions_root / "sequences" / self.sequence / "predictions"
        )

    def _ensure_output_dir(self) -> None:
        """Create output directory if needed."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def clear_output_dir(self) -> None:
        """Remove existing prediction labels for this sequence."""
        self._ensure_output_dir()
        for label_file in self.output_dir.glob("*.label"):
            label_file.unlink()

    def _output_path(self, frame_id: int) -> Path:
        """Get output path for a frame."""
        return self.output_dir / f"{frame_id:06d}.label"

    def _load_box_index(self) -> dict[int, list[dict]]:
        """Load box records once for efficient frame lookup."""
        if self._boxes_by_frame is not None:
            return self._boxes_by_frame

        self._boxes_by_frame = {}
        if not self.boxes_file.exists():
            return self._boxes_by_frame

        with open(self.boxes_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                frame_data = json.loads(line)
                frame_id = int(frame_data.get("frame_id", -1))
                if frame_id >= 0:
                    self._boxes_by_frame[frame_id] = frame_data.get("boxes", [])

        return self._boxes_by_frame

    def convert_frame(self, frame_id: int) -> ConversionStats | None:
        """Convert a single frame.

        Args:
            frame_id: Frame ID to convert.

        Returns:
            ConversionStats if successful.
        """
        # Load point cloud
        frame_data = self.dataset.get_frame(frame_id)
        points = frame_data.points

        # Load boxes for this frame
        boxes = self._load_box_index().get(frame_id)
        if boxes is None:
            boxes = []

        # Convert boxes to labels
        labels, stats = convert_boxes_to_labels(points, boxes, self.mapping)
        stats = ConversionStats(
            frame_id=frame_id,
            num_points=stats.num_points,
            num_boxes=stats.num_boxes,
            points_in_boxes=stats.points_in_boxes,
            points_outside_boxes=stats.points_outside_boxes,
            class_distribution=stats.class_distribution,
        )

        # Ensure output directory exists
        self._ensure_output_dir()

        # Write label file
        output_path = self._output_path(frame_id)
        write_label_file(output_path, labels)

        logger.debug(
            f"Frame {frame_id}: {stats.points_in_boxes}/{stats.num_points} points "
            f"in {stats.num_boxes} boxes"
        )

        return stats

    def convert_all_frames(
        self,
        start_frame: int = 0,
        end_frame: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[ConversionStats]:
        """Convert all frames in the sequence.

        Args:
            start_frame: First frame to convert (inclusive).
            end_frame: Last frame to convert (exclusive). None = all frames.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            List of ConversionStats for converted frames.
        """
        if end_frame is None:
            end_frame = self.dataset.num_frames

        num_frames = end_frame - start_frame
        results = []

        logger.info(
            f"Converting boxes to labels for sequence {self.sequence}, "
            f"frames {start_frame} to {end_frame - 1}"
        )

        for i, frame_id in enumerate(range(start_frame, end_frame)):
            stats = self.convert_frame(frame_id)
            if stats is not None:
                results.append(stats)

            if progress_callback:
                progress_callback(i + 1, num_frames)

        # Log summary
        total_points = sum(s.num_points for s in results)
        total_in_boxes = sum(s.points_in_boxes for s in results)
        coverage = total_in_boxes / total_points * 100 if total_points > 0 else 0

        logger.info(
            f"Conversion complete: {len(results)}/{num_frames} frames, "
            f"{total_in_boxes:,}/{total_points:,} points in boxes ({coverage:.1f}%)"
        )

        return results

    def get_available_frames(self) -> list[int]:
        """Get list of frame IDs that have box data.

        Returns:
            List of frame IDs with available boxes.
        """
        if not self.boxes_file.exists():
            return []

        frame_ids = []
        with open(self.boxes_file, "r", encoding="utf-8") as f:
            for line in f:
                frame_data = json.loads(line)
                frame_ids.append(frame_data.get("frame_id", -1))

        return sorted(frame_ids)
