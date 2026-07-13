"""Regression tests for distance-adaptive DBSCAN."""

import numpy as np
import pytest

from src.proposals.dbscan import cluster_dbscan_distance_adaptive
from src.utils.points import labels_to_clusters


@pytest.mark.parametrize(
    ("boundary", "expected_bin_inputs"),
    [
        (20.0, [3, 4, 0]),
        (40.0, [0, 3, 4]),
    ],
)
def test_distance_adaptive_dbscan_keeps_object_connected_across_bin_boundary(
    boundary: float,
    expected_bin_inputs: list[int],
) -> None:
    x_coordinates = np.array(
        [
            boundary - 0.3,
            boundary - 0.2,
            boundary - 0.1,
            boundary,
            boundary + 0.1,
            boundary + 0.2,
            boundary + 0.3,
        ],
        dtype=np.float32,
    )
    points = np.column_stack(
        (x_coordinates, np.zeros_like(x_coordinates), np.zeros_like(x_coordinates))
    )

    result = cluster_dbscan_distance_adaptive(
        points,
        distance_bins=[0.0, 20.0, 40.0, 60.0],
        bin_eps=[0.4, 0.6, 1.0],
        min_points=2,
    )

    assert result.num_clusters == 1
    assert np.all(result.labels == result.labels[0])
    assert result.labels[0] >= 0
    np.testing.assert_array_equal(result.clusters[0], np.arange(7))
    assert result.input_points == 7
    assert result.clustered_points == 7
    assert [bin_stats.input_points for bin_stats in result.bins or []] == expected_bin_inputs
    assert sum(bin_stats.input_points for bin_stats in result.bins or []) == 7
    assert sum(bin_stats.clustered_points for bin_stats in result.bins or []) == 7


@pytest.mark.parametrize("boundary", [20.0, 40.0])
def test_distance_adaptive_dbscan_merges_boundary_clusters_with_unique_global_labels(
    boundary: float,
) -> None:
    boundary_object = np.array(
        [
            boundary - 0.3,
            boundary - 0.2,
            boundary - 0.1,
            boundary,
            boundary + 0.1,
            boundary + 0.2,
            boundary + 0.3,
        ],
        dtype=np.float32,
    )
    separate_object = np.array(
        [boundary + 3.0, boundary + 3.1, boundary + 3.2], dtype=np.float32
    )
    x_coordinates = np.concatenate((boundary_object, separate_object))
    points = np.column_stack(
        (x_coordinates, np.zeros_like(x_coordinates), np.zeros_like(x_coordinates))
    )

    result = cluster_dbscan_distance_adaptive(
        points,
        distance_bins=[0.0, 20.0, 40.0, 60.0],
        bin_eps=[0.4, 0.6, 1.0],
        min_points=2,
    )

    assert result.num_clusters == 2
    assert len(np.unique(result.labels[result.labels >= 0])) == 2
    for cluster_id, cluster_indices in enumerate(result.clusters):
        np.testing.assert_array_equal(result.labels[cluster_indices], cluster_id)
    clustered_indices = np.concatenate(result.clusters)
    np.testing.assert_array_equal(np.sort(clustered_indices), np.arange(points.shape[0]))
    expected_clusters = [np.arange(7), np.arange(7, 10)]
    actual_clusters = labels_to_clusters(result.labels)
    for actual_cluster, expected_cluster in zip(actual_clusters, expected_clusters, strict=True):
        np.testing.assert_array_equal(actual_cluster, expected_cluster)


def test_distance_adaptive_dbscan_does_not_promote_borrowed_only_noise() -> None:
    points = np.array(
        [[19.45, 0.0, 0.0], [19.95, 0.0, 0.0]], dtype=np.float32
    )

    result = cluster_dbscan_distance_adaptive(
        points,
        distance_bins=[0.0, 20.0, 40.0],
        bin_eps=[0.4, 0.6],
        min_points=2,
    )

    assert result.num_clusters == 0
    np.testing.assert_array_equal(result.labels, np.array([-1, -1], dtype=np.int32))
    assert result.clusters == []
