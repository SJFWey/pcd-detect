"""Oriented Bounding Box (OBB) fitting utilities.

This module provides OBB fitting for 3D point clusters using PCA-based
orientation estimation and robust z-boundary computation with percentiles.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..utils.points import validate_points


@dataclass(frozen=True)
class OBBParams:
    """Oriented Bounding Box parameters.

    Attributes:
        center: (3,) Center of the box in world coordinates.
        extent: (3,) Half-extents along the local axes (length, width, height).
        rotation: (3, 3) Rotation matrix from local to world coordinates.
        yaw: Yaw angle (rotation around z-axis) in radians.
        corners: (8, 3) Corner points of the box in world coordinates.
    """

    center: NDArray[np.float64]
    extent: NDArray[np.float64]
    rotation: NDArray[np.float64]
    yaw: float
    corners: NDArray[np.float64]

    @property
    def volume(self) -> float:
        """Compute volume of the bounding box."""
        return float(8.0 * np.prod(self.extent))

    @property
    def dimensions(self) -> tuple[float, float, float]:
        """Return full dimensions (length, width, height)."""
        return (
            float(2.0 * self.extent[0]),
            float(2.0 * self.extent[1]),
            float(2.0 * self.extent[2]),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "center": self.center.tolist(),
            "extent": self.extent.tolist(),
            "rotation": self.rotation.tolist(),
            "yaw": self.yaw,
            "dimensions": list(self.dimensions),
            "volume": self.volume,
        }


@dataclass(frozen=True)
class OBBFitResult:
    """Result of OBB fitting for a single cluster.

    Attributes:
        cluster_id: Cluster identifier.
        params: Fitted OBB parameters, or None if fitting failed.
        num_points: Number of points in the cluster.
        valid: Whether the fit is valid (passed volume constraints).
        reason: Reason for invalid fit (if applicable).
    """

    cluster_id: int
    params: OBBParams | None
    num_points: int
    valid: bool
    reason: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "cluster_id": self.cluster_id,
            "num_points": self.num_points,
            "valid": self.valid,
        }
        if self.params is not None:
            result["box"] = self.params.to_dict()
        if self.reason:
            result["reason"] = self.reason
        return result


def _compute_pca_rotation(points_xy: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute 2D rotation matrix from PCA of XY coordinates.

    Returns a 3x3 rotation matrix with identity Z component.
    """
    if points_xy.shape[0] < 2:
        return np.eye(3, dtype=np.float64)

    # Center the points
    centroid = np.mean(points_xy, axis=0)
    centered = points_xy - centroid

    # Compute covariance matrix
    cov = np.cov(centered.T)
    if cov.ndim == 0:
        # Single point or all points identical
        return np.eye(3, dtype=np.float64)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 1] = -eigenvectors[:, 1]

    # Build 3x3 rotation matrix
    rotation = np.eye(3, dtype=np.float64)
    rotation[:2, :2] = eigenvectors

    return rotation


def _compute_yaw_from_rotation(rotation: NDArray[np.float64]) -> float:
    """Extract yaw angle from rotation matrix."""
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _compute_obb_corners(
    center: NDArray[np.float64],
    extent: NDArray[np.float64],
    rotation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the 8 corner points of an OBB.

    Corner ordering follows the convention:
    - Corners 0-3: bottom face (z_min)
    - Corners 4-7: top face (z_max)
    - Each face is ordered counter-clockwise when viewed from above
    """
    # Local corner offsets
    signs = np.array(
        [
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1],
        ],
        dtype=np.float64,
    )
    local_corners = signs * extent

    # Transform to world coordinates
    world_corners = (rotation @ local_corners.T).T + center

    return world_corners


def fit_obb(
    points: NDArray[np.float32],
    *,
    z_percentile: float = 5.0,
) -> OBBParams | None:
    """
    Fit an Oriented Bounding Box to a point cluster.

    The orientation is determined by PCA in the XY plane. The Z boundaries
    are computed using percentiles for robustness against outliers.

    Args:
        points: (N, 3+) Point cloud array.
        z_percentile: Percentile for robust Z boundary estimation.
                      Uses z_percentile and (100 - z_percentile) for
                      min and max Z respectively.

    Returns:
        OBBParams if fitting succeeded, None if insufficient points.
    """
    validate_points(points)

    if points.shape[0] < 3:
        return None

    xyz = points[:, :3].astype(np.float64, copy=False)

    # Compute orientation from XY PCA
    rotation = _compute_pca_rotation(xyz[:, :2])

    # Transform points to local coordinate system
    centroid_xy = np.mean(xyz[:, :2], axis=0)
    centered_xy = xyz[:, :2] - centroid_xy
    local_xy = (rotation[:2, :2].T @ centered_xy.T).T

    # Compute XY extents from local coordinates
    xy_min = np.min(local_xy, axis=0)
    xy_max = np.max(local_xy, axis=0)
    xy_extent = (xy_max - xy_min) / 2.0
    xy_center_local = (xy_min + xy_max) / 2.0

    # Transform local center back to world coordinates
    xy_center_world = rotation[:2, :2] @ xy_center_local + centroid_xy

    # Compute Z boundaries using percentiles for robustness
    z_values = xyz[:, 2]
    z_min = float(np.percentile(z_values, z_percentile))
    z_max = float(np.percentile(z_values, 100.0 - z_percentile))
    z_extent = (z_max - z_min) / 2.0
    z_center = (z_min + z_max) / 2.0

    # Assemble center and extent
    center = np.array(
        [xy_center_world[0], xy_center_world[1], z_center], dtype=np.float64
    )
    extent = np.array([xy_extent[0], xy_extent[1], z_extent], dtype=np.float64)

    # Ensure non-negative extents
    extent = np.maximum(extent, 1e-6)

    # Compute yaw angle
    yaw = _compute_yaw_from_rotation(rotation)

    # Compute corner points
    corners = _compute_obb_corners(center, extent, rotation)

    return OBBParams(
        center=center,
        extent=extent,
        rotation=rotation,
        yaw=yaw,
        corners=corners,
    )


def fit_obb_to_cluster(
    points: NDArray[np.float32],
    cluster_indices: NDArray[np.int64],
    cluster_id: int,
    *,
    z_percentile: float = 5.0,
    min_volume: float = 0.0,
    max_volume: float = float("inf"),
) -> OBBFitResult:
    """
    Fit an OBB to a specific cluster and validate constraints.

    Args:
        points: Full point cloud array.
        cluster_indices: Indices of points belonging to this cluster.
        cluster_id: Identifier for this cluster.
        z_percentile: Percentile for robust Z boundary estimation.
        min_volume: Minimum valid box volume (cubic meters).
        max_volume: Maximum valid box volume (cubic meters).

    Returns:
        OBBFitResult with fitted parameters and validation status.
    """
    validate_points(points)
    num_points = int(cluster_indices.shape[0])

    if num_points < 3:
        return OBBFitResult(
            cluster_id=cluster_id,
            params=None,
            num_points=num_points,
            valid=False,
            reason="insufficient_points",
        )

    cluster_points = points[cluster_indices]
    params = fit_obb(cluster_points, z_percentile=z_percentile)

    if params is None:
        return OBBFitResult(
            cluster_id=cluster_id,
            params=None,
            num_points=num_points,
            valid=False,
            reason="fit_failed",
        )

    # Validate volume constraints
    volume = params.volume
    if volume < min_volume:
        return OBBFitResult(
            cluster_id=cluster_id,
            params=params,
            num_points=num_points,
            valid=False,
            reason=f"volume_too_small ({volume:.3f} < {min_volume:.3f})",
        )
    if volume > max_volume:
        return OBBFitResult(
            cluster_id=cluster_id,
            params=params,
            num_points=num_points,
            valid=False,
            reason=f"volume_too_large ({volume:.3f} > {max_volume:.3f})",
        )

    return OBBFitResult(
        cluster_id=cluster_id,
        params=params,
        num_points=num_points,
        valid=True,
    )


def fit_obbs_to_clusters(
    points: NDArray[np.float32],
    clusters: Sequence[NDArray[np.int64]],
    *,
    z_percentile: float = 5.0,
    min_volume: float = 0.0,
    max_volume: float = float("inf"),
) -> list[OBBFitResult]:
    """
    Fit OBBs to multiple clusters.

    Args:
        points: Full point cloud array.
        clusters: List of index arrays, one per cluster.
        z_percentile: Percentile for robust Z boundary estimation.
        min_volume: Minimum valid box volume (cubic meters).
        max_volume: Maximum valid box volume (cubic meters).

    Returns:
        List of OBBFitResult, one per input cluster.
    """
    results = []
    for cluster_id, cluster_indices in enumerate(clusters):
        result = fit_obb_to_cluster(
            points,
            cluster_indices,
            cluster_id,
            z_percentile=z_percentile,
            min_volume=min_volume,
            max_volume=max_volume,
        )
        results.append(result)
    return results


def fit_obbs_from_config(
    points: NDArray[np.float32],
    clusters: Sequence[NDArray[np.int64]],
    config: Mapping[str, object],
) -> list[OBBFitResult]:
    """
    Fit OBBs to clusters using configuration dictionary.

    Config keys:
        z_percentile: Percentile for robust Z boundary (default: 5.0)
        min_box_volume: Minimum valid volume in cubic meters (default: 0.0)
        max_box_volume: Maximum valid volume in cubic meters (default: inf)

    Args:
        points: Full point cloud array.
        clusters: List of index arrays, one per cluster.
        config: Configuration dictionary.

    Returns:
        List of OBBFitResult, one per input cluster.
    """
    z_percentile = float(config.get("z_percentile", 5.0))
    min_volume = float(config.get("min_box_volume", 0.0))
    max_volume = float(config.get("max_box_volume", float("inf")))

    return fit_obbs_to_clusters(
        points,
        clusters,
        z_percentile=z_percentile,
        min_volume=min_volume,
        max_volume=max_volume,
    )


def points_in_obb(
    points: NDArray[np.floating],
    center: NDArray[np.floating],
    extent: NDArray[np.floating],
    yaw: float,
) -> NDArray[np.bool_]:
    """
    Check which points lie inside an Oriented Bounding Box.

    The OBB is defined by its center, half-extents, and yaw angle (rotation
    around Z-axis). Points are transformed to the local coordinate system
    of the box and checked against the axis-aligned bounds.

    Args:
        points: (N, 3+) Point cloud array.
        center: (3,) Center of the OBB in world coordinates.
        extent: (3,) Half-extents along the local axes (length/2, width/2, height/2).
        yaw: Yaw angle (rotation around z-axis) in radians.

    Returns:
        (N,) Boolean mask where True indicates the point is inside the OBB.
    """
    validate_points(points)

    xyz = points[:, :3].astype(np.float64, copy=False)
    center = np.asarray(center, dtype=np.float64)
    extent = np.asarray(extent, dtype=np.float64)

    # Build rotation matrix from yaw
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation = np.array(
        [[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]], dtype=np.float64
    )

    # Transform points to local coordinates
    # local = R^T @ (world - center)
    centered = xyz - center
    local = (rotation.T @ centered.T).T

    # Check if local coordinates are within [-extent, extent]
    inside = np.all(np.abs(local) <= extent, axis=1)

    return inside


def points_in_obbs(
    points: NDArray[np.floating],
    boxes: Sequence[dict],
    *,
    valid_only: bool = True,
) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
    """
    Assign points to the first OBB that contains them.

    For each point, finds the first box (in order) that contains it.
    Points not in any box are assigned box_id = -1.

    Args:
        points: (N, 3+) Point cloud array.
        boxes: List of box dicts with keys: 'center', 'extent', 'yaw', 'valid'.
        valid_only: If True, only consider boxes with 'valid' == True.

    Returns:
        box_ids: (N,) Array of box IDs for each point (-1 if not in any box).
        in_any_box: (N,) Boolean mask where True indicates the point is in some box.
    """
    validate_points(points)

    n_points = points.shape[0]
    box_ids = np.full(n_points, -1, dtype=np.int32)
    in_any_box = np.zeros(n_points, dtype=np.bool_)

    for box_idx, box in enumerate(boxes):
        # Skip invalid boxes if requested
        if valid_only and not box.get("valid", True):
            continue

        center = np.array(box["center"], dtype=np.float64)
        extent = np.array(box["extent"], dtype=np.float64)
        yaw = float(box["yaw"])

        # Find points inside this box
        inside = points_in_obb(points, center, extent, yaw)

        # Only assign points that haven't been assigned yet
        new_assignments = inside & ~in_any_box
        box_ids[new_assignments] = box_idx
        in_any_box |= inside

    return box_ids, in_any_box
