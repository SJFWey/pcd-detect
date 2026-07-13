"""Visualization and video export module.

Provides tools for:
- Point cloud rendering with matplotlib
- Bounding box overlays
- Ground segmentation visualization
- Cluster visualization
- Video export (MP4)

Dependencies (optional):
    Install with: uv sync --extra viz

Usage:
    from src.viz import PointCloudRenderer, VideoExporter, FrameData

    renderer = PointCloudRenderer()
    image = renderer.render_points(points)

    exporter = VideoExporter()
    exporter.export_sequence(frames, "output.mp4")
"""

from .colors import (
    SEMANTIC_KITTI_COLORS,
    THING_CLASSES,
    colorize_by_distance,
    colorize_by_height,
    generate_cluster_colors,
    instance_colors_to_float,
    semantic_colors_to_float,
)
from .renderer import CameraParams, PointCloudRenderer, RenderConfig
from .video_export import FrameData, VideoConfig, VideoExporter, create_frame_generator

__all__ = [
    # Renderer
    "PointCloudRenderer",
    "RenderConfig",
    "CameraParams",
    # Video Export
    "VideoExporter",
    "VideoConfig",
    "FrameData",
    "create_frame_generator",
    # Colors
    "SEMANTIC_KITTI_COLORS",
    "THING_CLASSES",
    "semantic_colors_to_float",
    "instance_colors_to_float",
    "generate_cluster_colors",
    "colorize_by_height",
    "colorize_by_distance",
]
