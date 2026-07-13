import numpy as np

from src.eval.detection_eval import BoundingBox3D
from src.preprocess.roi import ROIBounds
from tools.render_bev import _boxes_in_display_roi, _display_bounds


def _box(x: float, y: float) -> BoundingBox3D:
    return BoundingBox3D(
        center=np.array([x, y, 0.0]),
        dimensions=np.array([4.0, 2.0, 1.5]),
        yaw=0.0,
    )


def test_display_bounds_uses_display_roi_without_changing_processing_bounds() -> None:
    config = {
        "preprocess": {"roi": {"x_min": 0, "x_max": 70, "y_min": -40, "y_max": 40, "z_min": -3, "z_max": 3}},
        "visualization": {"display_roi": {"x_min": 0, "x_max": 35, "y_min": -12, "y_max": 12}},
    }

    assert _display_bounds(config) == ROIBounds(0.0, 35.0, -12.0, 12.0, -3.0, 3.0)


def test_overlay_keeps_only_boxes_centered_in_visible_roi() -> None:
    bounds = ROIBounds(0.0, 35.0, -12.0, 12.0, -3.0, 3.0)

    visible = _boxes_in_display_roi([_box(10.0, 0.0), _box(35.0, 12.0), _box(36.0, 0.0), _box(10.0, -13.0)], bounds)

    assert [(box.center[0], box.center[1]) for box in visible] == [(10.0, 0.0), (35.0, 12.0)]
