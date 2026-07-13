"""Common point-cloud utilities shared across modules."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_points(points: NDArray[np.floating]) -> None:
    """Validate that `points` is a 2D array with at least XYZ columns."""
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            f"points must be a 2D array with at least 3 columns, got shape={points.shape}"
        )


def labels_to_clusters(labels: NDArray[np.int32]) -> list[NDArray[np.int64]]:
    """Convert a (N,) label array to a list of per-cluster index arrays.

    Labels < 0 are treated as noise and excluded.
    """
    if labels.size == 0:
        return []

    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return []

    valid_indices = np.nonzero(valid_mask)[0]
    sorted_order = np.argsort(labels[valid_mask], kind="stable")
    sorted_indices = valid_indices[sorted_order]
    sorted_labels = labels[valid_mask][sorted_order]
    split_points = np.flatnonzero(np.diff(sorted_labels)) + 1
    return [
        cluster.astype(np.int64, copy=False)
        for cluster in np.split(sorted_indices, split_points)
    ]


def clusters_to_labels(
    num_points: int,
    clusters: list[NDArray[np.int64]],
) -> NDArray[np.int32]:
    """Convert a list of cluster index arrays back to a (N,) label array."""
    labels = np.full((num_points,), -1, dtype=np.int32)
    for cluster_id, cluster_indices in enumerate(clusters):
        labels[cluster_indices] = cluster_id
    return labels
