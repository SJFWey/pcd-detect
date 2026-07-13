"""Ground segmentation utilities using Open3D RANSAC."""

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from ..utils.points import validate_points


@dataclass(frozen=True)
class GroundSegmentationResult:
    """Result of ground segmentation."""

    plane_model: NDArray[np.float64] | None
    candidate_plane_model: NDArray[np.float64] | None
    ground_mask: NDArray[np.bool_]
    ground_ratio: float
    normal_z: float | None
    reject_reason: str | None
    input_points: int
    fit_points: int
    ground_points: NDArray[np.float32]
    nonground_points: NDArray[np.float32]


def _fit_plane(
    points: NDArray[np.float32],
    *,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
    random_seed: int | None,
) -> NDArray[np.float64] | None:
    if points.shape[0] < ransac_n:
        return None

    if random_seed is not None:
        o3d.utility.random.seed(int(random_seed))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    plane_model, _ = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    return np.asarray(plane_model, dtype=np.float64)


def _no_ground_result(
    points: NDArray[np.float32],
    *,
    fit_points: int,
    candidate_plane_model: NDArray[np.float64] | None,
    normal_z: float | None,
    reject_reason: str,
) -> GroundSegmentationResult:
    return GroundSegmentationResult(
        plane_model=None,
        candidate_plane_model=candidate_plane_model,
        ground_mask=np.zeros((points.shape[0],), dtype=bool),
        ground_ratio=0.0,
        normal_z=normal_z,
        reject_reason=reject_reason,
        input_points=int(points.shape[0]),
        fit_points=fit_points,
        ground_points=points[:0].astype(np.float32, copy=False),
        nonground_points=points.astype(np.float32, copy=False),
    )


def segment_ground(
    points: NDArray[np.float32],
    *,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
    z_min: float | None = None,
    z_max: float | None = None,
    min_normal_z: float = 0.9,
    random_seed: int | None = 0,
) -> GroundSegmentationResult:
    """
    Segment ground points using an RANSAC plane model.

    The plane is fit on an optional low-height band (z_min, z_max). The
    resulting plane is then evaluated against all points. ``min_normal_z`` is
    the minimum absolute z component of the normalized plane normal, in [0, 1].
    """
    validate_points(points)
    num_points = int(points.shape[0])
    if num_points == 0:
        return _no_ground_result(
            points,
            fit_points=0,
            candidate_plane_model=None,
            normal_z=None,
            reject_reason="insufficient_fit_points",
        )

    if z_min is not None or z_max is not None:
        z_min_val = -np.inf if z_min is None else float(z_min)
        z_max_val = np.inf if z_max is None else float(z_max)
        fit_mask = (points[:, 2] >= z_min_val) & (points[:, 2] <= z_max_val)
        fit_points = points[fit_mask]
    else:
        fit_points = points

    candidate_plane_model = _fit_plane(
        fit_points,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        random_seed=random_seed,
    )

    fit_point_count = int(fit_points.shape[0])
    if candidate_plane_model is None:
        return _no_ground_result(
            points,
            fit_points=fit_point_count,
            candidate_plane_model=None,
            normal_z=None,
            reject_reason="insufficient_fit_points",
        )

    a, b, c, d = candidate_plane_model
    norm = float(np.sqrt(a * a + b * b + c * c))
    if norm == 0.0:
        return _no_ground_result(
            points,
            fit_points=fit_point_count,
            candidate_plane_model=candidate_plane_model,
            normal_z=None,
            reject_reason="degenerate_plane",
        )

    normal_z = abs(c) / norm
    if normal_z < min_normal_z:
        return _no_ground_result(
            points,
            fit_points=fit_point_count,
            candidate_plane_model=candidate_plane_model,
            normal_z=normal_z,
            reject_reason="normal_constraint",
        )

    distances = np.abs(points[:, :3] @ candidate_plane_model[:3] + d) / norm
    ground_mask = distances <= distance_threshold
    ground_points = points[ground_mask].astype(np.float32, copy=False)
    nonground_points = points[~ground_mask].astype(np.float32, copy=False)
    ground_ratio = float(ground_points.shape[0]) / float(num_points)

    return GroundSegmentationResult(
        plane_model=candidate_plane_model,
        candidate_plane_model=candidate_plane_model,
        ground_mask=ground_mask,
        ground_ratio=ground_ratio,
        normal_z=normal_z,
        reject_reason=None,
        input_points=num_points,
        fit_points=fit_point_count,
        ground_points=ground_points,
        nonground_points=nonground_points,
    )
