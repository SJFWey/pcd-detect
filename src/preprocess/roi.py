"""ROI cropping utilities."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from ..utils.points import validate_points


@dataclass(frozen=True)
class ROIBounds:
    """Axis-aligned ROI bounds in KITTI Velodyne coordinates.

    KITTI uses x=forward, y=left, z=up in meters.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def validate(self) -> None:
        """Validate that each min bound is less than its max bound."""
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be smaller than x_max")
        if self.y_min >= self.y_max:
            raise ValueError("y_min must be smaller than y_max")
        if self.z_min >= self.z_max:
            raise ValueError("z_min must be smaller than z_max")

    @classmethod
    def from_config(cls, config: Mapping[str, float]) -> "ROIBounds":
        """Create ROIBounds from a config mapping."""
        required = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        missing = [key for key in required if key not in config]
        if missing:
            missing_text = ", ".join(missing)
            raise KeyError(f"Missing ROI bounds: {missing_text}")

        bounds = cls(
            x_min=float(config["x_min"]),
            x_max=float(config["x_max"]),
            y_min=float(config["y_min"]),
            y_max=float(config["y_max"]),
            z_min=float(config["z_min"]),
            z_max=float(config["z_max"]),
        )
        bounds.validate()
        return bounds


@dataclass(frozen=True)
class ROIResult:
    """Result of ROI cropping."""

    points: NDArray[np.float32]
    mask: NDArray[np.bool_]
    input_points: int
    output_points: int


def roi_mask(points: NDArray[np.float32], bounds: ROIBounds) -> NDArray[np.bool_]:
    """Compute ROI mask for input points."""
    validate_points(points)
    bounds.validate()
    xyz = points[:, :3]
    return (
        (xyz[:, 0] >= bounds.x_min)
        & (xyz[:, 0] <= bounds.x_max)
        & (xyz[:, 1] >= bounds.y_min)
        & (xyz[:, 1] <= bounds.y_max)
        & (xyz[:, 2] >= bounds.z_min)
        & (xyz[:, 2] <= bounds.z_max)
    )


def crop_points(points: NDArray[np.float32], bounds: ROIBounds) -> ROIResult:
    """Crop points by ROI bounds and return result with counts."""
    mask = roi_mask(points, bounds)
    cropped = points[mask].astype(np.float32, copy=False)
    return ROIResult(
        points=cropped,
        mask=mask,
        input_points=int(points.shape[0]),
        output_points=int(cropped.shape[0]),
    )


def apply_mask(
    mask: NDArray[np.bool_],
    *arrays: NDArray[np.generic],
) -> tuple[NDArray[np.generic], ...]:
    """Apply a boolean mask to one or more arrays with matching first dimension."""
    if not arrays:
        return ()

    expected = mask.shape[0]
    filtered = []
    for array in arrays:
        if array.shape[0] != expected:
            raise ValueError(
                "All arrays must share the same length as the mask. "
                f"Expected {expected}, got {array.shape[0]}"
            )
        filtered.append(array[mask])
    return tuple(filtered)
