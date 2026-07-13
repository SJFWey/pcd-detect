"""Interactive point cloud viewer using Open3D.

This module is used by the CLI `viz` command to provide an interactive
3D popup window instead of writing static PNGs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..boxes.obb import OBBParams
from ..utils.logging import get_logger
from .colors import (
    COLOR_BOX_DEFAULT,
    COLOR_GT_BOX,
    COLOR_GROUND,
    COLOR_NON_GROUND,
    colorize_by_height,
    generate_cluster_colors,
)
from .video_export import FrameData

logger = get_logger(__name__)


def _require_open3d():
    try:
        import open3d as o3d  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Open3D is required for interactive visualization. "
            "Install with: `uv sync` (or `uv pip install open3d`)."
        ) from e
    return o3d


def _subsample_indices(num_points: int, max_points: int | None, seed: int) -> NDArray[np.int64]:
    if max_points is None or num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_points, size=max_points, replace=False).astype(np.int64))


def _colors_points(points_xyz: NDArray[np.float32]) -> NDArray[np.float32]:
    return colorize_by_height(points_xyz)


def _colors_ground(
    points_xyz: NDArray[np.float32],
    ground_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    colors = np.zeros((len(points_xyz), 3), dtype=np.float32)
    colors[ground_mask] = COLOR_GROUND
    colors[~ground_mask] = COLOR_NON_GROUND
    return colors


def _colors_clusters(
    points_xyz: NDArray[np.float32],
    cluster_labels: NDArray[np.int32],
    seed: int,
) -> NDArray[np.float32]:
    _ = points_xyz  # points only needed for length validation
    if len(cluster_labels) == 0:
        return np.full((0, 3), 0.3, dtype=np.float32)

    max_label = int(cluster_labels.max())
    num_clusters = max(max_label + 1, 0)
    cluster_colors = generate_cluster_colors(max(num_clusters, 1), seed=seed)

    colors = np.full((len(cluster_labels), 3), 0.3, dtype=np.float32)  # gray for noise
    for i in range(num_clusters):
        mask = cluster_labels == i
        if np.any(mask):
            colors[mask] = cluster_colors[i % len(cluster_colors)]
    return colors


def _lineset_from_obb(
    o3d,
    box: OBBParams,
    rgb01: NDArray[np.float32],
):
    corners = np.asarray(box.corners, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError(f"OBB corners must have shape (8, 3), got {corners.shape}")

    # Edges follow the same convention as `PointCloudRenderer._draw_box`.
    edges = np.array(
        [
            # Bottom face
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            # Top face
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            # Vertical edges
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.int32,
    )

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(corners)
    lines.lines = o3d.utility.Vector2iVector(edges)
    color = np.asarray(rgb01, dtype=np.float64).reshape(1, 3)
    color = np.nan_to_num(color, nan=0.0, posinf=1.0, neginf=0.0)
    color = np.clip(color, 0.0, 1.0)
    lines.colors = o3d.utility.Vector3dVector(np.repeat(color, len(lines.lines), axis=0))
    return lines


@dataclass
class _ViewerState:
    mode: str
    boxes_visible: bool
    boxes_added: bool


def view_frame_open3d(
    frame_data: FrameData,
    *,
    mode: str = "debug",
    show_predictions: bool = True,
    show_gt: bool = True,
    max_points: int | None = 200_000,
    point_size: float = 2.0,
    seed: int = 0,
) -> None:
    """Open an interactive Open3D window for a single frame.

    Key bindings (when available):
      - 1: points (height color)
      - 2: ground (requires ground_mask)
      - 3: clusters (requires cluster_labels)
      - 4: boxes (pred/gt boxes)
      - B: toggle boxes overlay
    """
    o3d = _require_open3d()

    points_xyz = np.asarray(frame_data.points[:, :3], dtype=np.float32)
    ground_mask = frame_data.ground_mask
    cluster_labels = frame_data.cluster_labels

    # Filter out non-finite points (NaN/Inf) to avoid Open3D color warnings and
    # potential viewer issues.
    finite_mask = np.isfinite(points_xyz).all(axis=1)
    if not np.all(finite_mask):
        points_xyz = points_xyz[finite_mask]
        if ground_mask is not None:
            ground_mask = np.asarray(ground_mask, dtype=np.bool_)[finite_mask]
        if cluster_labels is not None:
            cluster_labels = np.asarray(cluster_labels, dtype=np.int32)[finite_mask]
        logger.info(
            f"Open3D viz: filtered non-finite points ({int(np.sum(~finite_mask))} removed)"
        )

    idx = _subsample_indices(len(points_xyz), max_points, seed=seed)
    points_xyz = points_xyz[idx]
    if ground_mask is not None:
        ground_mask = np.asarray(ground_mask, dtype=np.bool_)[idx]
    if cluster_labels is not None:
        cluster_labels = np.asarray(cluster_labels, dtype=np.int32)[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64, copy=False))

    pred_box_lines = []
    if show_predictions and frame_data.boxes:
        for box in frame_data.boxes:
            if (
                np.isfinite(box.center).all()
                and np.isfinite(box.rotation).all()
                and np.isfinite(box.extent).all()
                and np.isfinite(box.corners).all()
            ):
                pred_box_lines.append(_lineset_from_obb(o3d, box, COLOR_BOX_DEFAULT))

    gt_box_lines = []
    if show_gt and frame_data.gt_boxes:
        for box in frame_data.gt_boxes:
            if (
                np.isfinite(box.center).all()
                and np.isfinite(box.rotation).all()
                and np.isfinite(box.extent).all()
                and np.isfinite(box.corners).all()
            ):
                gt_box_lines.append(_lineset_from_obb(o3d, box, COLOR_GT_BOX))

    window_name = f"pcd-detect | frame={frame_data.frame_id} | mode={mode}"
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name, width=1280, height=720)

    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.background_color = np.asarray([0.05, 0.05, 0.05], dtype=np.float64)
        render_option.point_size = float(point_size)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis.add_geometry(coord, reset_bounding_box=True)
    vis.add_geometry(pcd, reset_bounding_box=True)

    state = _ViewerState(mode="points", boxes_visible=True, boxes_added=False)

    def set_boxes_visible(enable: bool) -> None:
        if enable and not state.boxes_added:
            for geom in pred_box_lines + gt_box_lines:
                vis.add_geometry(geom, reset_bounding_box=False)
            state.boxes_added = True
        elif (not enable) and state.boxes_added:
            for geom in pred_box_lines + gt_box_lines:
                vis.remove_geometry(geom, reset_bounding_box=False)
            state.boxes_added = False

    def set_mode(requested: str) -> str:
        actual = requested
        if requested == "ground" and ground_mask is None:
            actual = "points"
        if requested == "clusters" and cluster_labels is None:
            actual = "points"

        if actual == "points":
            colors = _colors_points(points_xyz)
        elif actual == "ground":
            colors = _colors_ground(points_xyz, ground_mask)  # type: ignore[arg-type]
        elif actual == "clusters":
            colors = _colors_clusters(points_xyz, cluster_labels, seed=seed)  # type: ignore[arg-type]
        elif actual == "boxes":
            colors = _colors_points(points_xyz)
        else:
            actual = "points"
            colors = _colors_points(points_xyz)

        colors = np.asarray(colors, dtype=np.float64)
        colors = np.nan_to_num(colors, nan=0.0, posinf=1.0, neginf=0.0)
        colors = np.clip(colors, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64, copy=False))
        vis.update_geometry(pcd)

        want_boxes = state.boxes_visible and (actual == "boxes" or show_predictions)
        have_boxes = len(pred_box_lines) > 0 or len(gt_box_lines) > 0
        if have_boxes:
            set_boxes_visible(want_boxes)

        state.mode = actual
        vis.update_renderer()
        return actual

    def _cb_set_mode(new_mode: str):
        def _cb(_vis) -> bool:
            actual = set_mode(new_mode)
            logger.info(
                f"Open3D viz: mode={actual} (ground={'yes' if ground_mask is not None else 'no'}, "
                f"clusters={'yes' if cluster_labels is not None else 'no'}, "
                f"boxes={'yes' if (frame_data.boxes or frame_data.gt_boxes) else 'no'})"
            )
            return False

        return _cb

    def _cb_toggle_boxes(_vis) -> bool:
        state.boxes_visible = not state.boxes_visible
        set_mode(state.mode)
        return False

    vis.register_key_callback(ord("1"), _cb_set_mode("points"))
    vis.register_key_callback(ord("2"), _cb_set_mode("ground"))
    vis.register_key_callback(ord("3"), _cb_set_mode("clusters"))
    vis.register_key_callback(ord("4"), _cb_set_mode("boxes"))
    vis.register_key_callback(ord("B"), _cb_toggle_boxes)

    # Initial mode
    if mode == "debug":
        start = "boxes" if (frame_data.boxes or frame_data.gt_boxes) else "points"
    else:
        start = mode
    set_mode(start)

    logger.info(
        "Open3D controls: mouse to rotate/zoom/pan; keys 1/2/3/4 to switch modes; B toggles boxes."
    )
    vis.run()
    vis.destroy_window()
