import numpy as np

from src.boxes.obb import fit_obb
from src.preprocess.ground import segment_ground
from src.preprocess.roi import ROIBounds, crop_points


def test_crop_points_uses_inclusive_bounds() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0, 0.1],
            [1.0, 1.0, 1.0, 0.2],
            [2.0, 0.0, 0.0, 0.3],
            [0.0, -2.0, 0.0, 0.4],
        ],
        dtype=np.float32,
    )
    bounds = ROIBounds(0.0, 1.0, -1.0, 1.0, -0.5, 1.0)

    result = crop_points(points, bounds)

    assert result.input_points == 4
    assert result.output_points == 2
    assert result.mask.tolist() == [True, True, False, False]


def test_segment_ground_handles_empty_input() -> None:
    points = np.empty((0, 4), dtype=np.float32)

    result = segment_ground(
        points,
        distance_threshold=0.2,
        ransac_n=3,
        num_iterations=10,
    )

    assert result.plane_model is None
    assert result.ground_points.shape == (0, 4)
    assert result.nonground_points.shape == (0, 4)


def test_fit_obb_returns_dimensions_for_simple_cluster() -> None:
    points = np.array(
        [
            [-1.0, -0.5, 0.0],
            [1.0, -0.5, 0.0],
            [1.0, 0.5, 1.0],
            [-1.0, 0.5, 1.0],
            [0.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )

    obb = fit_obb(points, z_percentile=0.0)

    assert obb is not None
    assert np.allclose(sorted(obb.dimensions[:2]), [1.0, 2.0], atol=1e-6)
    assert obb.dimensions[2] == 1.0
