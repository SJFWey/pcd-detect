"""BEV connected-components clustering for proposal generation."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..utils.points import clusters_to_labels, labels_to_clusters, validate_points


@dataclass(frozen=True)
class BEVCCStats:
    """Summary statistics for BEV connected-components clustering."""

    num_clusters: int
    mean_cluster_size: float
    occupied_cells: int


@dataclass(frozen=True)
class BEVCCResult:
    """Result of BEV connected-components clustering."""

    labels: NDArray[np.int32]
    clusters: list[NDArray[np.int64]]
    stats: BEVCCStats
    input_points: int
    clustered_points: int
    grid_origin: tuple[float, float]
    grid_shape: tuple[int, int]
    resolution: float


def cluster_bev_cc(
    points: NDArray[np.float32],
    *,
    resolution: float,
    min_height: float,
    min_points: int,
) -> BEVCCResult:
    """
    Cluster points using BEV occupancy and connected components.

    Points are projected to a 2D grid in the XY plane. A grid cell is
    considered occupied if the height range of points in that cell is
    at least `min_height`. Connected components are extracted on the
    occupancy grid and mapped back to 3D clusters.
    """
    validate_points(points)
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if min_height < 0:
        raise ValueError("min_height must be non-negative")
    if min_points < 1:
        raise ValueError("min_points must be at least 1")

    num_points = int(points.shape[0])
    if num_points == 0:
        empty_labels = np.zeros((0,), dtype=np.int32)
        stats = BEVCCStats(num_clusters=0, mean_cluster_size=0.0, occupied_cells=0)
        return BEVCCResult(
            labels=empty_labels,
            clusters=[],
            stats=stats,
            input_points=0,
            clustered_points=0,
            grid_origin=(0.0, 0.0),
            grid_shape=(0, 0),
            resolution=float(resolution),
        )

    xy = points[:, :2].astype(np.float64, copy=False)
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = max_xy - min_xy

    grid_width = int(np.floor(span[0] / resolution)) + 1
    grid_height = int(np.floor(span[1] / resolution)) + 1
    grid_shape = (grid_height, grid_width)

    coords = np.floor((xy - min_xy) / resolution).astype(np.int64)
    coords[:, 0] = np.clip(coords[:, 0], 0, grid_width - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, grid_height - 1)

    cell_ids = coords[:, 1] * grid_width + coords[:, 0]
    unique_cells, inverse = np.unique(cell_ids, return_inverse=True)

    z_vals = points[:, 2].astype(np.float64, copy=False)
    cell_min = np.full(unique_cells.shape[0], np.inf, dtype=np.float64)
    cell_max = np.full(unique_cells.shape[0], -np.inf, dtype=np.float64)
    np.minimum.at(cell_min, inverse, z_vals)
    np.maximum.at(cell_max, inverse, z_vals)
    cell_heights = cell_max - cell_min

    height_threshold = float(min_height)
    occupied_mask = cell_heights >= height_threshold
    occupied_cells = int(np.count_nonzero(occupied_mask))

    labels = np.full((num_points,), -1, dtype=np.int32)
    if occupied_cells > 0:
        occupied_ids = unique_cells[occupied_mask]
        occupied_y = occupied_ids // grid_width
        occupied_x = occupied_ids % grid_width

        grid = np.zeros(grid_shape, dtype=bool)
        grid[occupied_y, occupied_x] = True

        structure = np.ones((3, 3), dtype=np.int8)
        labeled_grid, _ = ndimage.label(grid, structure=structure)

        cell_labels = np.zeros(unique_cells.shape[0], dtype=np.int32)
        cell_labels[occupied_mask] = labeled_grid[occupied_y, occupied_x].astype(
            np.int32
        )
        labels = cell_labels[inverse].astype(np.int32) - 1

    clusters = labels_to_clusters(labels)
    if min_points > 1 and clusters:
        clusters = [cluster for cluster in clusters if cluster.size >= min_points]
        labels = clusters_to_labels(num_points, clusters)

    clustered_points = int(sum(cluster.size for cluster in clusters))
    num_clusters = len(clusters)
    mean_cluster_size = (
        float(clustered_points) / float(num_clusters) if num_clusters > 0 else 0.0
    )

    stats = BEVCCStats(
        num_clusters=num_clusters,
        mean_cluster_size=mean_cluster_size,
        occupied_cells=occupied_cells,
    )

    return BEVCCResult(
        labels=labels,
        clusters=clusters,
        stats=stats,
        input_points=num_points,
        clustered_points=clustered_points,
        grid_origin=(float(min_xy[0]), float(min_xy[1])),
        grid_shape=grid_shape,
        resolution=float(resolution),
    )


def cluster_bev_cc_from_config(
    points: NDArray[np.float32],
    config: Mapping[str, object],
) -> BEVCCResult:
    """Cluster points using BEV connected components and config settings."""
    resolution = float(config.get("resolution", 0.2))
    min_height = float(config.get("min_height", 0.0))
    min_points = int(config.get("min_points", 1))
    return cluster_bev_cc(
        points,
        resolution=resolution,
        min_height=min_height,
        min_points=min_points,
    )
