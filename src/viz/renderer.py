"""Point cloud renderer using matplotlib for offline rendering.

This module provides a matplotlib-based renderer for generating
static images and video frames of point clouds with bounding boxes.
Designed for headless rendering and video export.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
    semantic_colors_to_float,
)

logger = get_logger(__name__)


@dataclass
class CameraParams:
    """Camera parameters for 3D rendering.

    Attributes:
        elevation: Vertical angle in degrees (0 = horizontal, 90 = top-down).
        azimuth: Horizontal angle in degrees (0 = front view).
        distance: Distance from origin (for auto-scaling).
        center: Camera look-at point.
        fov: Field of view in degrees (for perspective).
    """

    elevation: float = 30.0
    azimuth: float = -60.0
    distance: float = 50.0
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov: float = 60.0


@dataclass
class RenderConfig:
    """Configuration for rendering.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        dpi: Dots per inch for matplotlib.
        point_size: Size of points.
        background_color: RGB background color.
        show_axes: Whether to show coordinate axes.
        show_grid: Whether to show ground grid.
        title: Optional title for the figure.
    """

    width: int = 1920
    height: int = 1080
    dpi: int = 100
    point_size: float = 1.0
    background_color: tuple[float, float, float] = (0.1, 0.1, 0.1)
    show_axes: bool = True
    show_grid: bool = True
    title: str | None = None


class PointCloudRenderer:
    """Matplotlib-based point cloud renderer for offline rendering.

    Supports:
    - Point clouds with custom colors
    - Bounding box overlays
    - Multiple camera views
    - Ground segmentation visualization
    - Cluster visualization
    """

    def __init__(self, config: RenderConfig | None = None):
        """Initialize renderer.

        Args:
            config: Rendering configuration. Uses defaults if None.
        """
        self.config = config or RenderConfig()
        self._fig = None
        self._ax = None

    def _setup_figure(self) -> None:
        """Set up matplotlib figure and 3D axes."""
        import matplotlib.pyplot as plt

        plt.switch_backend("Agg")  # Non-interactive backend

        figsize = (
            self.config.width / self.config.dpi,
            self.config.height / self.config.dpi,
        )
        self._fig = plt.figure(figsize=figsize, dpi=self.config.dpi)
        self._ax = self._fig.add_subplot(111, projection="3d")

        # Set background color
        self._fig.patch.set_facecolor(self.config.background_color)
        self._ax.set_facecolor(self.config.background_color)

        # Configure axes appearance
        if not self.config.show_axes:
            self._ax.set_axis_off()
        else:
            self._ax.set_xlabel("X (m)", color="white")
            self._ax.set_ylabel("Y (m)", color="white")
            self._ax.set_zlabel("Z (m)", color="white")
            self._ax.tick_params(colors="white")

    def _close_figure(self) -> None:
        """Close matplotlib figure."""
        import matplotlib.pyplot as plt

        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

    def _canvas_to_rgb(self) -> NDArray[np.uint8]:
        """Convert the current matplotlib canvas to an RGB image array.

        Uses `buffer_rgba` when available (matplotlib>=3.9 removed
        `tostring_rgb`), and falls back to `tostring_rgb` for older
        versions. Always returns (H, W, 3) uint8.
        """

        if self._fig is None:
            raise RuntimeError("Figure is not initialized")

        canvas = self._fig.canvas
        canvas.draw()  # Ensure renderer is ready
        width, height = canvas.get_width_height()

        if hasattr(canvas, "buffer_rgba"):
            buffer = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            image = buffer.reshape((height, width, 4))[..., :3]
        elif hasattr(canvas, "tostring_rgb"):
            buffer = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            image = buffer.reshape((height, width, 3))
        else:
            raise RuntimeError("Canvas does not support buffer_rgba or tostring_rgb")

        return image.copy()

    @staticmethod
    def _subsample_indices(num_points: int, max_points: int) -> NDArray[np.int64] | None:
        """Return deterministic subsample indices for stable renders."""
        if num_points <= max_points:
            return None
        return np.linspace(0, num_points - 1, max_points, dtype=np.int64)

    def _set_axis_limits(self, arrays: Sequence[NDArray[np.floating]]) -> None:
        """Set axis limits from non-empty point arrays."""
        if self._ax is None:
            raise RuntimeError("Axes are not initialized")

        non_empty = [array[:, :3] for array in arrays if array.shape[0] > 0]
        if not non_empty:
            self._ax.set_xlim([-1.0, 1.0])
            self._ax.set_ylim([-1.0, 1.0])
            self._ax.set_zlim([-1.0, 1.0])
            return

        all_pts = np.vstack(non_empty)
        margin = 5.0
        self._ax.set_xlim([all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin])
        self._ax.set_ylim([all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin])
        self._ax.set_zlim([all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin])

    def set_camera(self, camera: CameraParams) -> None:
        """Set camera view parameters.

        Args:
            camera: Camera parameters.
        """
        if self._ax is None:
            return
        self._ax.view_init(elev=camera.elevation, azim=camera.azimuth)

    def render_points(
        self,
        points: NDArray[np.float32],
        colors: NDArray[np.float32] | None = None,
        camera: CameraParams | None = None,
        title: str | None = None,
    ) -> NDArray[np.uint8]:
        """Render point cloud to image.

        Args:
            points: (N, 3+) point cloud coordinates.
            colors: (N, 3) RGB colors in [0, 1] range. Auto-colored by height if None.
            camera: Camera parameters. Uses defaults if None.
            title: Optional title override.

        Returns:
            (H, W, 3) RGB image array.
        """
        self._setup_figure()
        try:
            if self._ax is None:
                raise RuntimeError("Axes are not initialized")

            if colors is None:
                colors = colorize_by_height(points)

            # Subsample for performance if too many points
            max_points = 100000
            indices = self._subsample_indices(len(points), max_points)
            if indices is not None:
                points = points[indices]
                colors = colors[indices]

            if len(points) > 0:
                self._ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    c=colors,
                    s=self.config.point_size,
                    marker=".",
                )

            # Set camera
            cam = camera or CameraParams()
            self.set_camera(cam)

            self._set_axis_limits([points])

            # Title
            if title or self.config.title:
                self._ax.set_title(title or self.config.title, color="white")

            return self._canvas_to_rgb()
        finally:
            self._close_figure()

    def render_with_boxes(
        self,
        points: NDArray[np.float32],
        boxes: Sequence[OBBParams],
        colors: NDArray[np.float32] | None = None,
        box_color: NDArray[np.float32] | None = None,
        gt_boxes: Sequence[OBBParams] | None = None,
        camera: CameraParams | None = None,
        title: str | None = None,
    ) -> NDArray[np.uint8]:
        """Render point cloud with bounding box overlays.

        Args:
            points: (N, 3+) point cloud.
            boxes: Predicted bounding boxes.
            colors: Point colors.
            box_color: Color for predicted boxes.
            gt_boxes: Ground truth boxes (rendered in green).
            camera: Camera parameters.
            title: Optional title.

        Returns:
            (H, W, 3) RGB image array.
        """
        self._setup_figure()
        try:
            if self._ax is None:
                raise RuntimeError("Axes are not initialized")

            if colors is None:
                colors = colorize_by_height(points)

            # Subsample points
            max_points = 100000
            indices = self._subsample_indices(len(points), max_points)
            if indices is not None:
                plot_points = points[indices]
                plot_colors = colors[indices]
            else:
                plot_points = points
                plot_colors = colors

            if len(plot_points) > 0:
                self._ax.scatter(
                    plot_points[:, 0],
                    plot_points[:, 1],
                    plot_points[:, 2],
                    c=plot_colors,
                    s=self.config.point_size,
                    marker=".",
                    alpha=0.6,
                )

            # Draw predicted boxes
            pred_color = box_color if box_color is not None else COLOR_BOX_DEFAULT
            for box in boxes:
                self._draw_box(box, pred_color)

            # Draw GT boxes
            if gt_boxes:
                for box in gt_boxes:
                    self._draw_box(box, COLOR_GT_BOX)

            # Camera
            cam = camera or CameraParams()
            self.set_camera(cam)

            # Auto-scale axes
            all_points = [points]
            for box in boxes:
                all_points.append(box.corners)
            if gt_boxes:
                for box in gt_boxes:
                    all_points.append(box.corners)
            self._set_axis_limits(all_points)

            if title or self.config.title:
                self._ax.set_title(title or self.config.title, color="white")

            return self._canvas_to_rgb()
        finally:
            self._close_figure()

    def _draw_box(self, box: OBBParams, color: NDArray[np.float32]) -> None:
        """Draw a 3D bounding box."""
        corners = box.corners

        # Draw edges
        # Bottom face
        edges_bottom = [(0, 1), (1, 2), (2, 3), (3, 0)]
        # Top face
        edges_top = [(4, 5), (5, 6), (6, 7), (7, 4)]
        # Vertical edges
        edges_vertical = [(0, 4), (1, 5), (2, 6), (3, 7)]

        all_edges = edges_bottom + edges_top + edges_vertical

        for i, j in all_edges:
            self._ax.plot3D(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color=color,
                linewidth=2,
            )

    def render_ground_segmentation(
        self,
        points: NDArray[np.float32],
        ground_mask: NDArray[np.bool_],
        camera: CameraParams | None = None,
        title: str | None = None,
    ) -> NDArray[np.uint8]:
        """Render ground segmentation result.

        Args:
            points: (N, 3+) point cloud.
            ground_mask: (N,) boolean mask (True = ground).
            camera: Camera parameters.
            title: Optional title.

        Returns:
            (H, W, 3) RGB image array.
        """
        colors = np.zeros((len(points), 3), dtype=np.float32)
        colors[ground_mask] = COLOR_GROUND
        colors[~ground_mask] = COLOR_NON_GROUND

        return self.render_points(points, colors, camera, title)

    def render_clusters(
        self,
        points: NDArray[np.float32],
        cluster_labels: NDArray[np.int32],
        camera: CameraParams | None = None,
        title: str | None = None,
    ) -> NDArray[np.uint8]:
        """Render cluster visualization.

        Args:
            points: (N, 3+) point cloud.
            cluster_labels: (N,) cluster labels (-1 = noise).
            camera: Camera parameters.
            title: Optional title.

        Returns:
            (H, W, 3) RGB image array.
        """
        num_clusters = cluster_labels.max() + 1 if len(cluster_labels) > 0 else 0
        cluster_colors = generate_cluster_colors(max(num_clusters, 1))

        colors = np.full((len(points), 3), 0.3, dtype=np.float32)  # Gray for noise
        for i in range(num_clusters):
            mask = cluster_labels == i
            colors[mask] = cluster_colors[i % len(cluster_colors)]

        return self.render_points(points, colors, camera, title)

    def render_semantic(
        self,
        points: NDArray[np.float32],
        semantic_ids: NDArray[np.uint16],
        camera: CameraParams | None = None,
        title: str | None = None,
    ) -> NDArray[np.uint8]:
        """Render semantic segmentation.

        Args:
            points: (N, 3+) point cloud.
            semantic_ids: (N,) semantic class IDs.
            camera: Camera parameters.
            title: Optional title.

        Returns:
            (H, W, 3) RGB image array.
        """
        colors = semantic_colors_to_float(semantic_ids)
        return self.render_points(points, colors, camera, title)

    def save_image(
        self,
        image: NDArray[np.uint8],
        output_path: Path | str,
    ) -> None:
        """Save image to file.

        Args:
            image: (H, W, 3) RGB image array.
            output_path: Output file path (PNG, JPG, etc.).
        """
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.imsave(str(output_path), image)
        logger.info(f"Saved image to {output_path}")
