"""Cluster filtering utilities for proposal cleanup."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..utils.points import validate_points


@dataclass(frozen=True)
class ClusterFilterResult:
    """Result of cluster filtering."""

    clusters: list[NDArray[np.int64]]
    input_clusters: int
    output_clusters: int
    input_points: int
    output_points: int


def _validate_cluster_indices(
    points: NDArray[np.float32],
    cluster_indices: NDArray[np.int64],
) -> None:
    if cluster_indices.size == 0:
        return
    if np.any(cluster_indices < 0) or np.any(cluster_indices >= points.shape[0]):
        raise ValueError("Cluster indices must be within the points array bounds")


def filter_clusters(
    points: NDArray[np.float32],
    clusters: Sequence[NDArray[np.int64]],
    config: Mapping[str, object],
) -> ClusterFilterResult:
    """
    Filter clusters by size, height range, and percentile clipping.

    Percentile clipping trims outliers in XYZ independently using the same
    symmetric percentile on both ends (e.g., 2% and 98%).
    """
    validate_points(points)

    min_points = int(config.get("min_points", 1))
    max_points = int(config.get("max_points", 1_000_000_000))
    min_height = float(config.get("min_height", 0.0))
    max_height = float(config.get("max_height", float("inf")))
    percentile_clip = float(config.get("percentile_clip", 0.0))

    if min_points < 1:
        raise ValueError("min_points must be at least 1")
    if max_points < min_points:
        raise ValueError("max_points must be >= min_points")
    if min_height < 0:
        raise ValueError("min_height must be non-negative")
    if max_height < min_height:
        raise ValueError("max_height must be >= min_height")
    if percentile_clip < 0 or percentile_clip >= 50:
        raise ValueError("percentile_clip must be in [0, 50)")

    input_clusters = len(clusters)
    input_points = int(sum(cluster.size for cluster in clusters))
    filtered_clusters: list[NDArray[np.int64]] = []

    for cluster in clusters:
        cluster_indices = np.asarray(cluster, dtype=np.int64)
        if cluster_indices.size == 0:
            continue

        _validate_cluster_indices(points, cluster_indices)

        if cluster_indices.size < min_points or cluster_indices.size > max_points:
            continue

        cluster_points = points[cluster_indices]

        if percentile_clip > 0.0:
            lower = np.percentile(cluster_points[:, :3], percentile_clip, axis=0)
            upper = np.percentile(cluster_points[:, :3], 100.0 - percentile_clip, axis=0)
            mask = np.all(
                (cluster_points[:, :3] >= lower)
                & (cluster_points[:, :3] <= upper),
                axis=1,
            )
            cluster_indices = cluster_indices[mask]
            if cluster_indices.size == 0:
                continue
            cluster_points = points[cluster_indices]

        z_vals = cluster_points[:, 2]
        height = float(z_vals.max() - z_vals.min()) if z_vals.size > 0 else 0.0
        if height < min_height or height > max_height:
            continue

        if cluster_indices.size < min_points or cluster_indices.size > max_points:
            continue

        filtered_clusters.append(cluster_indices)

    output_points = int(sum(cluster.size for cluster in filtered_clusters))
    return ClusterFilterResult(
        clusters=filtered_clusters,
        input_clusters=input_clusters,
        output_clusters=len(filtered_clusters),
        input_points=input_points,
        output_points=output_points,
    )
