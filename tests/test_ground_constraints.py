import numpy as np

from src.preprocess.ground import segment_ground


def _vertical_wall() -> np.ndarray:
    x_values = np.linspace(-2.0, 2.0, 11, dtype=np.float32)
    z_values = np.linspace(-1.8, -0.8, 9, dtype=np.float32)
    x_grid, z_grid = np.meshgrid(x_values, z_values)
    return np.column_stack(
        (
            x_grid.ravel(),
            np.zeros(x_grid.size, dtype=np.float32),
            z_grid.ravel(),
        )
    ).astype(np.float32)


def test_segment_ground_rejects_vertical_ransac_candidate() -> None:
    points = _vertical_wall()

    result = segment_ground(
        points,
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=100,
        z_min=-2.0,
        z_max=-0.5,
        min_normal_z=0.95,
    )

    assert result.plane_model is None
    assert not result.ground_mask.any()
    assert result.reject_reason == "normal_constraint"
    assert result.candidate_plane_model is not None
    candidate_normal = result.candidate_plane_model[:3]
    candidate_normal_z = abs(candidate_normal[2]) / np.linalg.norm(candidate_normal)
    assert candidate_normal_z < 0.05


def test_segment_ground_accepts_horizontal_plane_with_diagnostics() -> None:
    axis = np.linspace(-2.0, 2.0, 7, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(axis, axis)
    ground = np.column_stack(
        (
            x_grid.ravel(),
            y_grid.ravel(),
            np.full(x_grid.size, -1.0, dtype=np.float32),
        )
    )
    elevated = np.array(
        [[-1.0, -1.0, 0.5], [1.0, -1.0, 0.7], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    points = np.vstack((ground, elevated)).astype(np.float32)

    result = segment_ground(
        points,
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=100,
        z_min=-1.2,
        z_max=-0.8,
        min_normal_z=0.95,
    )

    assert result.plane_model is not None
    assert result.candidate_plane_model is not None
    assert result.ground_mask[: ground.shape[0]].all()
    assert not result.ground_mask[ground.shape[0] :].any()
    assert result.normal_z is not None
    assert result.normal_z > 0.99
    assert result.ground_ratio == ground.shape[0] / points.shape[0]
    assert result.reject_reason is None


def test_segment_ground_reports_insufficient_fit_points() -> None:
    points = np.array(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 0.5],
            [1.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )

    result = segment_ground(
        points,
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=100,
        z_min=-1.1,
        z_max=-0.9,
    )

    assert result.plane_model is None
    assert result.candidate_plane_model is None
    assert not result.ground_mask.any()
    assert result.ground_ratio == 0.0
    assert result.fit_points == 2
    assert result.nonground_points.shape == points.shape
    assert result.reject_reason == "insufficient_fit_points"
