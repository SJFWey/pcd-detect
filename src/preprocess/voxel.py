"""Voxel downsampling utilities."""

from dataclasses import dataclass
from typing import Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from ..utils.points import validate_points


@dataclass(frozen=True)
class VoxelBinStats:
    """Statistics for a distance bin in adaptive voxel downsampling."""

    bin_index: int
    distance_min: float
    distance_max: float
    voxel_size: float
    input_points: int
    output_points: int


@dataclass(frozen=True)
class VoxelDownsampleResult:
    """Result of voxel downsampling."""

    points: NDArray[np.float32]
    input_points: int
    output_points: int
    bins: list[VoxelBinStats] | None = None


def _voxel_downsample(
    points: NDArray[np.float32], voxel_size: float
) -> NDArray[np.float32]:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    if points.shape[0] == 0:
        return points.astype(np.float32, copy=False)

    coords = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    _, inverse = np.unique(coords, axis=0, return_inverse=True)

    counts = np.bincount(inverse)
    sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
    for dim in range(3):
        sums[:, dim] = np.bincount(inverse, weights=points[:, dim])

    centroids = (sums / counts[:, None]).astype(np.float32)
    return centroids


def voxel_downsample_fixed(
    points: NDArray[np.float32], voxel_size: float
) -> VoxelDownsampleResult:
    """Downsample points using a fixed voxel size."""
    validate_points(points)
    downsampled = _voxel_downsample(points, voxel_size)
    return VoxelDownsampleResult(
        points=downsampled,
        input_points=int(points.shape[0]),
        output_points=int(downsampled.shape[0]),
    )


def voxel_downsample_distance_adaptive(
    points: NDArray[np.float32],
    distance_bins: Sequence[float],
    voxel_sizes: Sequence[float],
) -> VoxelDownsampleResult:
    """
    Downsample points with distance-adaptive voxel sizes.

    Distances are computed in the XY plane. Points outside the provided
    bin range are clamped to the nearest bin.
    """
    validate_points(points)
    if len(distance_bins) < 2:
        raise ValueError("distance_bins must have at least two edges")

    if len(voxel_sizes) != len(distance_bins) - 1:
        raise ValueError(
            "voxel_sizes length must be len(distance_bins) - 1, "
            f"got {len(voxel_sizes)} and {len(distance_bins)}"
        )

    edges = np.asarray(distance_bins, dtype=np.float32)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("distance_bins must be strictly increasing")

    sizes = np.asarray(voxel_sizes, dtype=np.float32)
    if np.any(sizes <= 0):
        raise ValueError("voxel_sizes must all be positive")

    if points.shape[0] == 0:
        return VoxelDownsampleResult(
            points=points.astype(np.float32, copy=False),
            input_points=0,
            output_points=0,
            bins=[
                VoxelBinStats(
                    bin_index=i,
                    distance_min=float(edges[i]),
                    distance_max=float(edges[i + 1]),
                    voxel_size=float(sizes[i]),
                    input_points=0,
                    output_points=0,
                )
                for i in range(len(sizes))
            ],
        )

    distances = np.linalg.norm(points[:, :2], axis=1)
    bin_indices = np.digitize(distances, edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, len(sizes) - 1)

    downsampled_chunks: list[NDArray[np.float32]] = []
    bin_stats: list[VoxelBinStats] = []

    for i in range(len(sizes)):
        mask = bin_indices == i
        bin_points = points[mask]
        if bin_points.shape[0] == 0:
            downsampled = bin_points.astype(np.float32, copy=False)
        else:
            downsampled = _voxel_downsample(bin_points, float(sizes[i]))
        downsampled_chunks.append(downsampled)
        bin_stats.append(
            VoxelBinStats(
                bin_index=i,
                distance_min=float(edges[i]),
                distance_max=float(edges[i + 1]),
                voxel_size=float(sizes[i]),
                input_points=int(bin_points.shape[0]),
                output_points=int(downsampled.shape[0]),
            )
        )

    merged = (
        np.concatenate(downsampled_chunks, axis=0)
        if downsampled_chunks
        else points.astype(np.float32, copy=False)
    )
    return VoxelDownsampleResult(
        points=merged,
        input_points=int(points.shape[0]),
        output_points=int(merged.shape[0]),
        bins=bin_stats,
    )


def downsample_points(
    points: NDArray[np.float32],
    config: Mapping[str, object],
) -> VoxelDownsampleResult:
    """Downsample points according to a config dictionary."""
    validate_points(points)
    enabled = bool(config.get("enabled", True))
    if not enabled:
        return VoxelDownsampleResult(
            points=points.astype(np.float32, copy=False),
            input_points=int(points.shape[0]),
            output_points=int(points.shape[0]),
        )

    mode = str(config.get("mode", "fixed"))
    if mode == "fixed":
        voxel_size = float(cast(float, config.get("voxel_size", 0.1)))
        return voxel_downsample_fixed(points, voxel_size)

    if mode == "distance_adaptive":
        distance_bins = cast(Sequence[float], config.get("distance_bins", []))
        voxel_sizes = cast(Sequence[float], config.get("bin_voxel_sizes", []))
        return voxel_downsample_distance_adaptive(
            points,
            distance_bins=distance_bins,
            voxel_sizes=voxel_sizes,
        )

    raise ValueError(f"Unsupported voxel mode: {mode}")
