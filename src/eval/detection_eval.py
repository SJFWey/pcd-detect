"""Detection evaluation metrics using SemanticKITTI instance IDs.

This module provides evaluation tools for 3D object detection:
- Ground truth instance extraction from SemanticKITTI labels
- IoU computation (BEV and 3D)
- Hungarian matching between predictions and ground truth
- Precision, Recall, F1 metrics
- Distance-stratified evaluation

Usage:
    evaluator = DetectionEvaluator(iou_threshold=0.5)

    for frame in frames:
        gt_boxes = extract_gt_instances(frame.points, frame.inst_label, frame.sem_label)
        evaluator.add_frame(pred_boxes, gt_boxes, frame_id)

    metrics = evaluator.compute_metrics()
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall: {metrics.recall:.3f}")
    print(f"F1: {metrics.f1:.3f}")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from ..boxes.obb import OBBParams, fit_obb
from ..datasets.semkitti_labels import SemanticKITTILabels
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Distance bins for stratified evaluation (in meters)
DEFAULT_DISTANCE_BINS = [0, 10, 20, 30, 40, 50, float("inf")]

# Canonical thing-class groups for matching static and moving SemanticKITTI ids.
CANONICAL_SEMANTIC_IDS = {
    252: 10,  # moving-car
    253: 31,  # moving-bicyclist
    254: 30,  # moving-person
    255: 32,  # moving-motorcyclist
    256: 15,  # moving-on-rails
    257: 13,  # moving-bus
    258: 18,  # moving-truck
    259: 20,  # moving-other-vehicle
}


@dataclass(frozen=True)
class BoundingBox3D:
    """3D bounding box representation for evaluation.

    Attributes:
        center: (3,) Center coordinates [x, y, z].
        dimensions: (3,) Full dimensions [length, width, height].
        yaw: Rotation angle around z-axis in radians.
        instance_id: Instance identifier (for GT boxes).
        semantic_id: Semantic class ID.
        score: Confidence score (for predictions).
        num_points: Number of points in the box.
    """

    center: NDArray[np.float64]
    dimensions: NDArray[np.float64]
    yaw: float
    instance_id: int = 0
    semantic_id: int = 0
    score: float = 1.0
    num_points: int = 0

    @property
    def volume(self) -> float:
        """Compute box volume."""
        return float(np.prod(self.dimensions))

    @property
    def distance(self) -> float:
        """Compute distance from origin (typically sensor position)."""
        return float(np.sqrt(self.center[0] ** 2 + self.center[1] ** 2))

    @classmethod
    def from_obb_params(
        cls,
        params: OBBParams,
        instance_id: int = 0,
        semantic_id: int = 0,
        score: float = 1.0,
        num_points: int = 0,
    ) -> "BoundingBox3D":
        """Create from OBBParams."""
        return cls(
            center=params.center.copy(),
            dimensions=np.array(params.dimensions, dtype=np.float64),
            yaw=params.yaw,
            instance_id=instance_id,
            semantic_id=semantic_id,
            score=score,
            num_points=num_points,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "center": self.center.tolist(),
            "dimensions": self.dimensions.tolist(),
            "yaw": self.yaw,
            "instance_id": self.instance_id,
            "semantic_id": self.semantic_id,
            "score": self.score,
            "num_points": self.num_points,
            "volume": self.volume,
            "distance": self.distance,
        }


def canonical_semantic_id(semantic_id: int) -> int:
    """Map moving SemanticKITTI thing ids to their static class ids."""
    return CANONICAL_SEMANTIC_IDS.get(int(semantic_id), int(semantic_id))


def semantic_ids_match(pred_id: int, gt_id: int) -> bool:
    """Return True when prediction and GT classes are evaluation-compatible."""
    pred_canonical = canonical_semantic_id(pred_id)
    gt_canonical = canonical_semantic_id(gt_id)
    return pred_canonical != 0 and pred_canonical == gt_canonical


def _rectangle_corners_xy(box: BoundingBox3D) -> NDArray[np.float64]:
    """Return BEV rectangle corners in counter-clockwise order."""
    length = float(box.dimensions[0])
    width = float(box.dimensions[1])
    if length <= 0.0 or width <= 0.0:
        return np.empty((0, 2), dtype=np.float64)

    half_length = length / 2.0
    half_width = width / 2.0
    local = np.array(
        [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )
    cos_yaw = np.cos(box.yaw)
    sin_yaw = np.sin(box.yaw)
    rotation = np.array(
        [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64
    )
    return (rotation @ local.T).T + box.center[:2]


def _polygon_area(poly: NDArray[np.float64]) -> float:
    """Compute area of a 2D polygon with the shoelace formula."""
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _is_inside_clip_edge(
    point: NDArray[np.float64],
    edge_start: NDArray[np.float64],
    edge_end: NDArray[np.float64],
) -> bool:
    """Return True if point is on the inside of a CCW clip edge."""
    edge = edge_end - edge_start
    rel = point - edge_start
    return bool(edge[0] * rel[1] - edge[1] * rel[0] >= -1e-9)


def _line_intersection(
    line_start: NDArray[np.float64],
    line_end: NDArray[np.float64],
    edge_start: NDArray[np.float64],
    edge_end: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Intersect two 2D lines, returning line_start for near-parallel lines."""
    line_vec = line_end - line_start
    edge_vec = edge_end - edge_start
    denom = line_vec[0] * edge_vec[1] - line_vec[1] * edge_vec[0]
    if abs(float(denom)) < 1e-12:
        return line_start

    delta = edge_start - line_start
    t = (delta[0] * edge_vec[1] - delta[1] * edge_vec[0]) / denom
    return line_start + t * line_vec


def _clip_polygon(
    subject: NDArray[np.float64],
    clip: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Clip a convex polygon by another convex polygon."""
    if subject.shape[0] == 0 or clip.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    output = subject.copy()
    for i in range(clip.shape[0]):
        edge_start = clip[i]
        edge_end = clip[(i + 1) % clip.shape[0]]
        input_poly = output
        if input_poly.shape[0] == 0:
            break

        clipped_points = []
        previous = input_poly[-1]
        previous_inside = _is_inside_clip_edge(previous, edge_start, edge_end)
        for current in input_poly:
            current_inside = _is_inside_clip_edge(current, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    clipped_points.append(
                        _line_intersection(previous, current, edge_start, edge_end)
                    )
                clipped_points.append(current)
            elif previous_inside:
                clipped_points.append(
                    _line_intersection(previous, current, edge_start, edge_end)
                )
            previous = current
            previous_inside = current_inside

        output = (
            np.array(clipped_points, dtype=np.float64)
            if clipped_points
            else np.empty((0, 2), dtype=np.float64)
        )

    return output


def compute_bev_intersection_area(box1: BoundingBox3D, box2: BoundingBox3D) -> float:
    """Compute BEV intersection area between two yawed boxes."""
    rect1 = _rectangle_corners_xy(box1)
    rect2 = _rectangle_corners_xy(box2)
    intersection = _clip_polygon(rect1, rect2)
    return _polygon_area(intersection)


def compute_bev_iou(box1: BoundingBox3D, box2: BoundingBox3D) -> float:
    """Compute Bird's Eye View (BEV) IoU between two boxes.

    Projects yawed boxes to XY plane and computes rotated rectangle IoU.

    Args:
        box1: First bounding box.
        box2: Second bounding box.

    Returns:
        IoU value in [0, 1].
    """
    inter_area = compute_bev_intersection_area(box1, box2)
    area1 = float(box1.dimensions[0] * box1.dimensions[1])
    area2 = float(box2.dimensions[0] * box2.dimensions[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return float(inter_area / union_area)


def compute_3d_iou(box1: BoundingBox3D, box2: BoundingBox3D) -> float:
    """Compute 3D IoU between two boxes.

    Uses rotated BEV intersection area and vertical overlap.

    Args:
        box1: First bounding box.
        box2: Second bounding box.

    Returns:
        IoU value in [0, 1].
    """
    bev_inter_area = compute_bev_intersection_area(box1, box2)
    if bev_inter_area <= 0.0:
        return 0.0

    z1_min = float(box1.center[2] - box1.dimensions[2] / 2.0)
    z1_max = float(box1.center[2] + box1.dimensions[2] / 2.0)
    z2_min = float(box2.center[2] - box2.dimensions[2] / 2.0)
    z2_max = float(box2.center[2] + box2.dimensions[2] / 2.0)
    inter_height = max(0.0, min(z1_max, z2_max) - max(z1_min, z2_min))
    inter_vol = bev_inter_area * inter_height

    vol1 = box1.volume
    vol2 = box2.volume
    union_vol = vol1 + vol2 - inter_vol

    if union_vol <= 0:
        return 0.0

    return inter_vol / union_vol


def compute_iou_matrix(
    pred_boxes: Sequence[BoundingBox3D],
    gt_boxes: Sequence[BoundingBox3D],
    use_3d_iou: bool = False,
    match_classes: bool = True,
) -> NDArray[np.float64]:
    """Compute IoU matrix between predictions and ground truth.

    Args:
        pred_boxes: List of predicted boxes.
        gt_boxes: List of ground truth boxes.
        use_3d_iou: If True, use 3D IoU; otherwise use BEV IoU.
        match_classes: If True, only same semantic classes receive non-zero IoU.

    Returns:
        (num_pred, num_gt) IoU matrix.
    """
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)

    if num_pred == 0 or num_gt == 0:
        return np.zeros((num_pred, num_gt), dtype=np.float64)

    iou_func = compute_3d_iou if use_3d_iou else compute_bev_iou
    iou_matrix = np.zeros((num_pred, num_gt), dtype=np.float64)

    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            if match_classes and not semantic_ids_match(
                pred.semantic_id, gt.semantic_id
            ):
                continue
            iou_matrix[i, j] = iou_func(pred, gt)

    return iou_matrix


class InstanceMatcher:
    """Match predicted boxes to ground truth using Hungarian algorithm.

    Uses IoU-based cost matrix and linear sum assignment to find
    optimal matching that maximizes total IoU.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        use_3d_iou: bool = False,
        match_classes: bool = True,
    ):
        """Initialize matcher.

        Args:
            iou_threshold: Minimum IoU to consider a match valid.
            use_3d_iou: If True, use 3D IoU; otherwise use BEV IoU.
            match_classes: If True, require matching semantic classes.
        """
        self.iou_threshold = iou_threshold
        self.use_3d_iou = use_3d_iou
        self.match_classes = match_classes

    def match(
        self,
        pred_boxes: Sequence[BoundingBox3D],
        gt_boxes: Sequence[BoundingBox3D],
    ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
        """Match predictions to ground truth.

        Args:
            pred_boxes: List of predicted boxes.
            gt_boxes: List of ground truth boxes.

        Returns:
            Tuple of:
            - matches: List of (pred_idx, gt_idx, iou) tuples for matched pairs.
            - unmatched_preds: List of prediction indices with no match.
            - unmatched_gts: List of ground truth indices with no match.
        """
        if len(pred_boxes) == 0:
            return [], [], list(range(len(gt_boxes)))
        if len(gt_boxes) == 0:
            return [], list(range(len(pred_boxes))), []

        # Compute IoU matrix
        iou_matrix = compute_iou_matrix(
            pred_boxes,
            gt_boxes,
            self.use_3d_iou,
            self.match_classes,
        )

        # Make valid edges cheaper than any combination of IoU improvements can
        # compensate for. This makes the Hungarian objective lexicographic:
        # maximum valid-match cardinality, then maximum summed IoU.
        valid_edges = np.isfinite(iou_matrix) & (iou_matrix >= self.iou_threshold)
        if self.match_classes:
            class_compatible = np.array(
                [
                    [semantic_ids_match(pred.semantic_id, gt.semantic_id) for gt in gt_boxes]
                    for pred in pred_boxes
                ],
                dtype=bool,
            )
            valid_edges &= class_compatible

        cardinality_priority = min(len(pred_boxes), len(gt_boxes)) + 1
        cost_matrix = np.zeros_like(iou_matrix)
        cost_matrix[valid_edges] = -(
            cardinality_priority + iou_matrix[valid_edges]
        )
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

        matches = []
        matched_preds = set()
        matched_gts = set()

        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            iou = iou_matrix[pred_idx, gt_idx]
            # Keep the threshold check as a defensive final guard.
            if valid_edges[pred_idx, gt_idx] and iou >= self.iou_threshold:
                matches.append((int(pred_idx), int(gt_idx), float(iou)))
                matched_preds.add(pred_idx)
                matched_gts.add(gt_idx)

        unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
        unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gts]

        return matches, unmatched_preds, unmatched_gts


@dataclass
class FrameResult:
    """Evaluation results for a single frame."""

    frame_id: int
    num_predictions: int
    num_ground_truth: int
    num_true_positives: int
    num_false_positives: int
    num_false_negatives: int
    matches: list[tuple[int, int, float]]
    avg_iou: float


@dataclass
class DetectionMetrics:
    """Aggregated detection metrics.

    Attributes:
        total_predictions: Total number of predicted boxes.
        total_ground_truth: Total number of ground truth boxes.
        true_positives: Number of correctly matched predictions.
        false_positives: Number of unmatched predictions.
        false_negatives: Number of unmatched ground truth boxes.
        precision: TP / (TP + FP).
        recall: TP / (TP + FN).
        f1: Harmonic mean of precision and recall.
        mean_iou: Average IoU of matched pairs.
        iou_histogram: Distribution of IoU values.
    """

    total_predictions: int = 0
    total_ground_truth: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mean_iou: float = 0.0
    iou_histogram: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_predictions": self.total_predictions,
            "total_ground_truth": self.total_ground_truth,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_iou": self.mean_iou,
            "iou_histogram": self.iou_histogram,
        }


class DetectionEvaluator:
    """Evaluator for 3D object detection.

    Accumulates frame-by-frame results and computes aggregate metrics.
    Supports distance-stratified evaluation.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        use_3d_iou: bool = False,
        distance_bins: list[float] | None = None,
        match_classes: bool = True,
    ):
        """Initialize evaluator.

        Args:
            iou_threshold: Minimum IoU to consider a match.
            use_3d_iou: If True, use 3D IoU; otherwise use BEV IoU.
            distance_bins: Distance bins for stratified evaluation.
            match_classes: If True, require matching semantic classes.
        """
        self.iou_threshold = iou_threshold
        self.use_3d_iou = use_3d_iou
        self.distance_bins = distance_bins or DEFAULT_DISTANCE_BINS
        self.match_classes = match_classes

        self.matcher = InstanceMatcher(iou_threshold, use_3d_iou, match_classes)

        # Accumulate results
        self._frame_results: list[FrameResult] = []
        self._all_ious: list[float] = []

        # Distance-stratified accumulators
        self._distance_tp: dict[str, int] = {}
        self._distance_fp: dict[str, int] = {}
        self._distance_fn: dict[str, int] = {}
        self._distance_ious: dict[str, list[float]] = {}

        self._init_distance_bins()

    def _init_distance_bins(self) -> None:
        """Initialize distance bin accumulators."""
        for i in range(len(self.distance_bins) - 1):
            bin_name = f"{self.distance_bins[i]:.0f}-{self.distance_bins[i + 1]:.0f}m"
            self._distance_tp[bin_name] = 0
            self._distance_fp[bin_name] = 0
            self._distance_fn[bin_name] = 0
            self._distance_ious[bin_name] = []

    def _get_distance_bin(self, distance: float) -> str | None:
        """Get distance bin name for a given distance."""
        for i in range(len(self.distance_bins) - 1):
            if self.distance_bins[i] <= distance < self.distance_bins[i + 1]:
                return f"{self.distance_bins[i]:.0f}-{self.distance_bins[i + 1]:.0f}m"
        return None

    def reset(self) -> None:
        """Reset all accumulated results."""
        self._frame_results.clear()
        self._all_ious.clear()
        self._init_distance_bins()

    def add_frame(
        self,
        pred_boxes: Sequence[BoundingBox3D],
        gt_boxes: Sequence[BoundingBox3D],
        frame_id: int = -1,
    ) -> FrameResult:
        """Add evaluation results for a single frame.

        Args:
            pred_boxes: Predicted bounding boxes.
            gt_boxes: Ground truth bounding boxes.
            frame_id: Frame identifier.

        Returns:
            FrameResult with per-frame metrics.
        """
        matches, unmatched_preds, unmatched_gts = self.matcher.match(
            pred_boxes, gt_boxes
        )

        num_tp = len(matches)
        num_fp = len(unmatched_preds)
        num_fn = len(unmatched_gts)

        # Compute average IoU for matches
        match_ious = [iou for _, _, iou in matches]
        avg_iou = float(np.mean(match_ious)) if match_ious else 0.0
        self._all_ious.extend(match_ious)

        # Update distance-stratified statistics
        # For matched predictions
        for pred_idx, gt_idx, iou in matches:
            gt_box = gt_boxes[gt_idx]
            bin_name = self._get_distance_bin(gt_box.distance)
            if bin_name:
                self._distance_tp[bin_name] += 1
                self._distance_ious[bin_name].append(iou)

        # For false positives
        for pred_idx in unmatched_preds:
            pred_box = pred_boxes[pred_idx]
            bin_name = self._get_distance_bin(pred_box.distance)
            if bin_name:
                self._distance_fp[bin_name] += 1

        # For false negatives
        for gt_idx in unmatched_gts:
            gt_box = gt_boxes[gt_idx]
            bin_name = self._get_distance_bin(gt_box.distance)
            if bin_name:
                self._distance_fn[bin_name] += 1

        result = FrameResult(
            frame_id=frame_id,
            num_predictions=len(pred_boxes),
            num_ground_truth=len(gt_boxes),
            num_true_positives=num_tp,
            num_false_positives=num_fp,
            num_false_negatives=num_fn,
            matches=matches,
            avg_iou=avg_iou,
        )
        self._frame_results.append(result)

        return result

    def compute_metrics(self) -> DetectionMetrics:
        """Compute aggregated detection metrics.

        Returns:
            DetectionMetrics with overall statistics.
        """
        if not self._frame_results:
            return DetectionMetrics()

        total_pred = sum(r.num_predictions for r in self._frame_results)
        total_gt = sum(r.num_ground_truth for r in self._frame_results)
        total_tp = sum(r.num_true_positives for r in self._frame_results)
        total_fp = sum(r.num_false_positives for r in self._frame_results)
        total_fn = sum(r.num_false_negatives for r in self._frame_results)

        precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        mean_iou = float(np.mean(self._all_ious)) if self._all_ious else 0.0

        # Compute IoU histogram
        iou_histogram = {}
        if self._all_ious:
            bins = [0.0, 0.25, 0.5, 0.75, 1.0]
            counts, _ = np.histogram(self._all_ious, bins=bins)
            for i, count in enumerate(counts):
                bin_name = f"{bins[i]:.2f}-{bins[i + 1]:.2f}"
                iou_histogram[bin_name] = int(count)

        return DetectionMetrics(
            total_predictions=total_pred,
            total_ground_truth=total_gt,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            precision=precision,
            recall=recall,
            f1=f1,
            mean_iou=mean_iou,
            iou_histogram=iou_histogram,
        )

    def compute_distance_metrics(self) -> dict[str, DetectionMetrics]:
        """Compute metrics stratified by distance.

        Returns:
            Dict mapping distance bin names to DetectionMetrics.
        """
        results = {}

        for bin_name in self._distance_tp.keys():
            tp = self._distance_tp[bin_name]
            fp = self._distance_fp[bin_name]
            fn = self._distance_fn[bin_name]
            ious = self._distance_ious[bin_name]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            mean_iou = float(np.mean(ious)) if ious else 0.0

            results[bin_name] = DetectionMetrics(
                total_predictions=tp + fp,
                total_ground_truth=tp + fn,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                mean_iou=mean_iou,
            )

        return results

    def get_frame_results(self) -> list[FrameResult]:
        """Get all frame-level results."""
        return self._frame_results.copy()

    def save_results(self, output_path: str | Path) -> None:
        """Save evaluation results to JSON file.

        Args:
            output_path: Path to output JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics = self.compute_metrics()
        distance_metrics = self.compute_distance_metrics()

        results = {
            "overall": metrics.to_dict(),
            "by_distance": {k: v.to_dict() for k, v in distance_metrics.items()},
            "config": {
                "iou_threshold": self.iou_threshold,
                "use_3d_iou": self.use_3d_iou,
                "match_classes": self.match_classes,
                "distance_bins": self.distance_bins,
            },
            "num_frames": len(self._frame_results),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved detection metrics to {output_path}")


def extract_gt_instances(
    points: NDArray[np.float32],
    instance_labels: NDArray[np.uint32],
    semantic_labels: NDArray[np.uint32],
    labels_helper: SemanticKITTILabels | None = None,
    min_points: int = 10,
) -> list[BoundingBox3D]:
    """Extract ground truth instance bounding boxes from SemanticKITTI labels.

    Args:
        points: (N, 3+) Point cloud array.
        instance_labels: (N,) Instance IDs (0 = no instance).
        semantic_labels: (N,) Semantic class IDs.
        labels_helper: Optional SemanticKITTILabels for class filtering.
        min_points: Minimum points per instance.

    Returns:
        List of BoundingBox3D for each valid instance.
    """
    if labels_helper is None:
        labels_helper = SemanticKITTILabels()

    gt_boxes = []

    # Get unique non-zero instance IDs
    unique_instances = np.unique(instance_labels)
    unique_instances = unique_instances[unique_instances > 0]

    for inst_id in unique_instances:
        mask = instance_labels == inst_id
        inst_points = points[mask]
        inst_semantic = semantic_labels[mask]

        if len(inst_points) < min_points:
            continue

        # Get dominant semantic class for this instance
        semantic_id = int(np.bincount(inst_semantic.astype(np.int64)).argmax())

        # Only include "thing" classes (objects with instances)
        if not labels_helper.is_thing(semantic_id):
            continue

        # Fit OBB to instance points
        obb_params = fit_obb(inst_points[:, :3])
        if obb_params is None:
            continue

        gt_box = BoundingBox3D.from_obb_params(
            obb_params,
            instance_id=int(inst_id),
            semantic_id=semantic_id,
            num_points=len(inst_points),
        )
        gt_boxes.append(gt_box)

    return gt_boxes
