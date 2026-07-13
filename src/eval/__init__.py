"""src.eval - Evaluation modules for detection and performance metrics."""

from .detection_eval import (
    DetectionEvaluator,
    DetectionMetrics,
    InstanceMatcher,
    compute_3d_iou,
    compute_bev_iou,
)
from .perf_report import (
    PerformanceReport,
    generate_performance_report,
)

__all__ = [
    "DetectionEvaluator",
    "DetectionMetrics",
    "InstanceMatcher",
    "compute_3d_iou",
    "compute_bev_iou",
    "PerformanceReport",
    "generate_performance_report",
]
