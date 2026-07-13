"""DBSCAN clustering utilities for proposal generation."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from ..utils.points import labels_to_clusters, validate_points


@dataclass(frozen=True)
class DBSCANBinStats:
    """Statistics for a single distance bin in adaptive DBSCAN."""

    bin_index: int
    distance_min: float
    distance_max: float
    eps: float
    input_points: int
    num_clusters: int
    clustered_points: int


@dataclass(frozen=True)
class DBSCANResult:
    """Result of DBSCAN clustering."""

    labels: NDArray[np.int32]
    clusters: list[NDArray[np.int64]]
    input_points: int
    clustered_points: int
    num_clusters: int
    mean_cluster_size: float
    mode: str
    eps: float | None = None
    bins: list[DBSCANBinStats] | None = None


def _cluster_dbscan(
    points: NDArray[np.float32],
    *,
    eps: float,
    min_points: int,
) -> NDArray[np.int32]:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        points[:, :3].astype(np.float64, copy=False)
    )
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    return labels.astype(np.int32, copy=False)


def cluster_dbscan_fixed(
    points: NDArray[np.float32],
    *,
    eps: float,
    min_points: int,
) -> DBSCANResult:
    """Cluster points using fixed-parameter DBSCAN."""
    validate_points(points)
    if eps <= 0:
        raise ValueError("eps must be positive")
    if min_points < 1:
        raise ValueError("min_points must be at least 1")

    num_points = int(points.shape[0])
    labels = _cluster_dbscan(points, eps=eps, min_points=min_points)
    clusters = labels_to_clusters(labels)
    clustered_points = int(sum(cluster.size for cluster in clusters))
    num_clusters = len(clusters)
    mean_cluster_size = (
        float(clustered_points) / float(num_clusters) if num_clusters > 0 else 0.0
    )

    return DBSCANResult(
        labels=labels,
        clusters=clusters,
        input_points=num_points,
        clustered_points=clustered_points,
        num_clusters=num_clusters,
        mean_cluster_size=mean_cluster_size,
        mode="fixed",
        eps=float(eps),
    )


def cluster_dbscan_distance_adaptive(
    points: NDArray[np.float32],
    *,
    distance_bins: Sequence[float],
    bin_eps: Sequence[float],
    min_points: int | Sequence[int],
) -> DBSCANResult:
    """Cluster points using distance-adaptive DBSCAN parameters.

    Args:
        points: (N, 3+) point cloud array.
        distance_bins: Distance bin edges in meters (e.g., [0, 20, 40, 60]).
        bin_eps: Epsilon values for each bin.
        min_points: Minimum points for clustering. Can be:
            - int: same value for all bins
            - Sequence[int]: per-bin values (length must match number of bins)
    """
    validate_points(points)
    if len(distance_bins) < 2:
        raise ValueError("distance_bins must have at least two edges")
    if len(bin_eps) != len(distance_bins) - 1:
        raise ValueError(
            "bin_eps length must be len(distance_bins) - 1, "
            f"got {len(bin_eps)} and {len(distance_bins)}"
        )

    num_bins = len(distance_bins) - 1

    # Handle min_points: can be int or per-bin sequence
    if isinstance(min_points, int):
        if min_points < 1:
            raise ValueError("min_points must be at least 1")
        bin_min_points = [min_points] * num_bins
    else:
        if len(min_points) != num_bins:
            raise ValueError(
                f"bin_min_points length must be {num_bins}, got {len(min_points)}"
            )
        if any(mp < 1 for mp in min_points):
            raise ValueError("all min_points values must be at least 1")
        bin_min_points = list(min_points)

    edges = np.asarray(distance_bins, dtype=np.float32)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("distance_bins must be strictly increasing")

    eps_values = np.asarray(bin_eps, dtype=np.float32)
    if np.any(eps_values <= 0):
        raise ValueError("bin_eps must all be positive")

    num_points = int(points.shape[0])
    labels = np.full((num_points,), -1, dtype=np.int32)
    if num_points == 0:
        return DBSCANResult(
            labels=labels,
            clusters=[],
            input_points=0,
            clustered_points=0,
            num_clusters=0,
            mean_cluster_size=0.0,
            mode="distance_adaptive",
            bins=[
                DBSCANBinStats(
                    bin_index=i,
                    distance_min=float(edges[i]),
                    distance_max=float(edges[i + 1]),
                    eps=float(eps_values[i]),
                    input_points=0,
                    num_clusters=0,
                    clustered_points=0,
                )
                for i in range(len(eps_values))
            ],
        )

    distances = np.linalg.norm(points[:, :2], axis=1)
    bin_indices = np.digitize(distances, edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, len(eps_values) - 1)

    local_clusters: list[NDArray[np.int64]] = []
    for i in range(len(eps_values)):
        owned_mask = bin_indices == i
        overlap_mask = owned_mask.copy()
        if i > 0:
            overlap_mask |= (bin_indices == i - 1) & (
                distances >= edges[i] - eps_values[i]
            )
        if i < len(eps_values) - 1:
            overlap_mask |= (bin_indices == i + 1) & (
                distances <= edges[i + 1] + eps_values[i]
            )

        local_indices = np.flatnonzero(overlap_mask).astype(np.int64, copy=False)
        local_labels = _cluster_dbscan(
            points[local_indices],
            eps=float(eps_values[i]),
            min_points=bin_min_points[i],
        )
        for local_label in np.unique(local_labels[local_labels >= 0]):
            local_cluster = local_indices[local_labels == local_label]
            # Borrowed points may bridge an owned cluster across a boundary, but
            # cannot independently create one under a neighboring bin's epsilon.
            if np.any(bin_indices[local_cluster] == i):
                local_clusters.append(local_cluster)

    parent = list(range(len(local_clusters)))

    def find(cluster_index: int) -> int:
        while parent[cluster_index] != cluster_index:
            parent[cluster_index] = parent[parent[cluster_index]]
            cluster_index = parent[cluster_index]
        return cluster_index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    cluster_by_point: dict[int, int] = {}
    for cluster_index, local_cluster in enumerate(local_clusters):
        for point_index in local_cluster:
            point = int(point_index)
            previous_cluster = cluster_by_point.setdefault(point, cluster_index)
            union(previous_cluster, cluster_index)

    component_members: dict[int, list[NDArray[np.int64]]] = {}
    for cluster_index, local_cluster in enumerate(local_clusters):
        component_members.setdefault(find(cluster_index), []).append(local_cluster)

    clusters = sorted(
        (
            np.unique(np.concatenate(members)).astype(np.int64, copy=False)
            for members in component_members.values()
        ),
        key=lambda cluster: int(cluster[0]),
    )
    labels = np.full((num_points,), -1, dtype=np.int32)
    for cluster_id, cluster in enumerate(clusters):
        labels[cluster] = cluster_id

    # Per-bin inputs and clustered points use canonical ownership, so both totals
    # remain additive. A final merged cluster can touch multiple bins, therefore
    # per-bin num_clusters is intentionally not additive.
    bin_stats = [
        DBSCANBinStats(
            bin_index=i,
            distance_min=float(edges[i]),
            distance_max=float(edges[i + 1]),
            eps=float(eps_values[i]),
            input_points=int(np.count_nonzero(bin_indices == i)),
            num_clusters=sum(
                bool(np.any(bin_indices[cluster] == i)) for cluster in clusters
            ),
            clustered_points=int(
                np.count_nonzero((bin_indices == i) & (labels >= 0))
            ),
        )
        for i in range(len(eps_values))
    ]
    clustered_points = int(sum(cluster.size for cluster in clusters))
    num_clusters = len(clusters)
    mean_cluster_size = (
        float(clustered_points) / float(num_clusters) if num_clusters > 0 else 0.0
    )

    return DBSCANResult(
        labels=labels,
        clusters=clusters,
        input_points=num_points,
        clustered_points=clustered_points,
        num_clusters=num_clusters,
        mean_cluster_size=mean_cluster_size,
        mode="distance_adaptive",
        bins=bin_stats,
    )


def cluster_dbscan_from_config(
    points: NDArray[np.float32],
    config: Mapping[str, object],
) -> DBSCANResult:
    """Cluster points with DBSCAN using a config mapping."""
    mode = str(config.get("mode", "fixed"))
    min_points = int(config.get("min_points", 10))

    if mode == "fixed":
        eps = float(config.get("eps", 0.5))
        return cluster_dbscan_fixed(points, eps=eps, min_points=min_points)

    if mode == "distance_adaptive":
        distance_bins = config.get("distance_bins", [])
        bin_eps = config.get("bin_eps", [])
        # Support per-bin min_points (bin_min_points) or fallback to global min_points
        bin_min_points = config.get("bin_min_points")
        if bin_min_points is None:
            # Use global min_points for all bins
            actual_min_points = min_points
        else:
            actual_min_points = [int(mp) for mp in bin_min_points]
        return cluster_dbscan_distance_adaptive(
            points,
            distance_bins=distance_bins,
            bin_eps=bin_eps,
            min_points=actual_min_points,
        )

    raise ValueError(f"Unsupported DBSCAN mode: {mode}")
