"""Video export utilities for point cloud visualization.

Generates MP4 videos from:
- Single frame debug views
- Full sequence playback
- Rotating camera views
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from ..boxes.obb import OBBParams
from ..utils.logging import get_logger
from .renderer import CameraParams, PointCloudRenderer, RenderConfig

logger = get_logger(__name__)


@dataclass
class VideoConfig:
    """Video export configuration.

    Attributes:
        fps: Frames per second.
        codec: Video codec (h264, libx264, etc.).
        quality: Quality setting (0-10, higher = better).
        width: Video width in pixels.
        height: Video height in pixels.
    """

    fps: int = 10
    codec: str = "libx264"
    quality: int = 8
    width: int = 1920
    height: int = 1080


@dataclass
class FrameData:
    """Data for a single video frame.

    Attributes:
        frame_id: Frame number/identifier.
        points: Point cloud array.
        colors: Optional point colors.
        boxes: Predicted bounding boxes.
        gt_boxes: Ground truth boxes.
        ground_mask: Ground segmentation mask.
        cluster_labels: Cluster labels.
        title: Frame title/label.
    """

    frame_id: int
    points: NDArray[np.float32]
    colors: NDArray[np.float32] | None = None
    boxes: Sequence[OBBParams] = field(default_factory=list)
    gt_boxes: Sequence[OBBParams] = field(default_factory=list)
    ground_mask: NDArray[np.bool_] | None = None
    cluster_labels: NDArray[np.int32] | None = None
    title: str | None = None


class VideoExporter:
    """Export point cloud visualizations to video.

    Supports multiple rendering modes:
    - debug: Full pipeline visualization per frame
    - sequence: Sequence playback from fixed viewpoint
    - rotating: 360-degree rotating view of single frame
    """

    def __init__(
        self,
        video_config: VideoConfig | None = None,
        render_config: RenderConfig | None = None,
    ):
        """Initialize video exporter.

        Args:
            video_config: Video encoding settings.
            render_config: Rendering settings.
        """
        self.video_config = video_config or VideoConfig()
        self.render_config = render_config or RenderConfig(
            width=self.video_config.width,
            height=self.video_config.height,
        )
        self.renderer = PointCloudRenderer(self.render_config)

    def export_sequence(
        self,
        frames: Iterator[FrameData] | Sequence[FrameData],
        output_path: Path | str,
        camera: CameraParams | None = None,
        mode: str = "boxes",
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Export a sequence of frames to video.

        Args:
            frames: Iterator or sequence of frame data.
            output_path: Output video file path.
            camera: Fixed camera view for all frames.
            mode: Rendering mode - 'points', 'boxes', 'ground', 'clusters'.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            Path to the exported video file.
        """
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio is required for video export. "
                "Install with: uv pip install imageio imageio-ffmpeg"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting video to {output_path}")

        cam = camera or CameraParams()

        # Convert to list if iterator (for length)
        if not isinstance(frames, Sequence):
            frames = list(frames)

        total_frames = len(frames)

        writer = imageio.get_writer(
            str(output_path),
            fps=self.video_config.fps,
            codec=self.video_config.codec,
            quality=self.video_config.quality,
        )

        try:
            for i, frame_data in enumerate(frames):
                # Render frame based on mode
                image = self._render_frame(frame_data, cam, mode)
                writer.append_data(image)

                if progress_callback:
                    progress_callback(i + 1, total_frames)

                if (i + 1) % 50 == 0:
                    logger.info(f"  Rendered {i + 1}/{total_frames} frames")

        finally:
            writer.close()

        logger.info(f"Video exported to {output_path}")
        return output_path

    def export_debug_video(
        self,
        frame_data: FrameData,
        output_path: Path | str,
        num_views: int = 4,
        frames_per_view: int = 30,
    ) -> Path:
        """Export a debug video showing multiple views of a single frame.

        Creates a video that rotates around the scene showing:
        1. Point cloud with height coloring
        2. Ground segmentation (if available)
        3. Cluster visualization (if available)
        4. Bounding boxes

        Args:
            frame_data: Frame data to visualize.
            output_path: Output video file path.
            num_views: Number of distinct camera angles.
            frames_per_view: Frames to hold each view.

        Returns:
            Path to the exported video file.
        """
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio is required for video export. "
                "Install with: uv pip install imageio imageio-ffmpeg"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting debug video to {output_path}")

        # Generate camera views (rotating around the scene)
        azimuths = np.linspace(-180, 180, num_views * frames_per_view, endpoint=False)
        elevations = np.ones_like(azimuths) * 30  # Fixed elevation

        writer = imageio.get_writer(
            str(output_path),
            fps=self.video_config.fps,
            codec=self.video_config.codec,
            quality=self.video_config.quality,
        )

        # Determine what data is available
        modes = ["points"]
        if frame_data.ground_mask is not None:
            modes.append("ground")
        if frame_data.cluster_labels is not None:
            modes.append("clusters")
        if frame_data.boxes or frame_data.gt_boxes:
            modes.append("boxes")

        try:
            for i, (azim, elev) in enumerate(zip(azimuths, elevations)):
                cam = CameraParams(elevation=elev, azimuth=azim)

                # Cycle through modes
                mode = modes[(i // (frames_per_view // len(modes))) % len(modes)]
                title = f"Frame {frame_data.frame_id} | {mode.upper()} | Az={azim:.0f}°"

                image = self._render_frame(frame_data, cam, mode, title_override=title)
                writer.append_data(image)

        finally:
            writer.close()

        logger.info(f"Debug video exported to {output_path}")
        return output_path

    def export_rotating_view(
        self,
        frame_data: FrameData,
        output_path: Path | str,
        mode: str = "boxes",
        duration_seconds: float = 10.0,
        elevation: float = 30.0,
    ) -> Path:
        """Export a 360-degree rotating view of a single frame.

        Args:
            frame_data: Frame data to visualize.
            output_path: Output video file path.
            mode: Rendering mode.
            duration_seconds: Video duration.
            elevation: Camera elevation angle.

        Returns:
            Path to the exported video file.
        """
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio is required for video export. "
                "Install with: uv pip install imageio imageio-ffmpeg"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_frames = int(duration_seconds * self.video_config.fps)
        azimuths = np.linspace(0, 360, total_frames, endpoint=False)

        writer = imageio.get_writer(
            str(output_path),
            fps=self.video_config.fps,
            codec=self.video_config.codec,
            quality=self.video_config.quality,
        )

        try:
            for i, azim in enumerate(azimuths):
                cam = CameraParams(elevation=elevation, azimuth=azim)
                title = f"Frame {frame_data.frame_id} | {mode}"
                image = self._render_frame(frame_data, cam, mode, title_override=title)
                writer.append_data(image)

        finally:
            writer.close()

        logger.info(f"Rotating view video exported to {output_path}")
        return output_path

    def _render_frame(
        self,
        frame_data: FrameData,
        camera: CameraParams,
        mode: str,
        title_override: str | None = None,
    ) -> NDArray[np.uint8]:
        """Render a single frame in the specified mode.

        Args:
            frame_data: Frame data.
            camera: Camera parameters.
            mode: Rendering mode.
            title_override: Override title.

        Returns:
            Rendered image array.
        """
        title = title_override or frame_data.title

        match mode:
            case "points":
                return self.renderer.render_points(
                    frame_data.points,
                    frame_data.colors,
                    camera,
                    title,
                )
            case "ground" if frame_data.ground_mask is not None:
                return self.renderer.render_ground_segmentation(
                    frame_data.points,
                    frame_data.ground_mask,
                    camera,
                    title,
                )
            case "clusters" if frame_data.cluster_labels is not None:
                return self.renderer.render_clusters(
                    frame_data.points,
                    frame_data.cluster_labels,
                    camera,
                    title,
                )
            case "boxes":
                return self.renderer.render_with_boxes(
                    frame_data.points,
                    list(frame_data.boxes),
                    frame_data.colors,
                    gt_boxes=list(frame_data.gt_boxes) if frame_data.gt_boxes else None,
                    camera=camera,
                    title=title,
                )
            case _:
                # Default to points
                return self.renderer.render_points(
                    frame_data.points,
                    frame_data.colors,
                    camera,
                    title,
                )

    def generate_thumbnail(
        self,
        frame_data: FrameData,
        output_path: Path | str,
        mode: str = "boxes",
        camera: CameraParams | None = None,
    ) -> Path:
        """Generate a thumbnail image for a frame.

        Args:
            frame_data: Frame data.
            output_path: Output image path.
            mode: Rendering mode.
            camera: Camera parameters.

        Returns:
            Path to saved thumbnail.
        """
        output_path = Path(output_path)
        cam = camera or CameraParams()

        image = self._render_frame(frame_data, cam, mode)
        self.renderer.save_image(image, output_path)

        return output_path


def create_frame_generator(
    dataset,
    pipeline_func: Callable,
    start_frame: int = 0,
    end_frame: int | None = None,
    step: int = 1,
) -> Iterator[FrameData]:
    """Create a frame generator for video export.

    This is a helper function to generate FrameData objects from
    a dataset and processing pipeline.

    Args:
        dataset: KITTI dataset object.
        pipeline_func: Function that processes a frame and returns
                      (boxes, ground_mask, cluster_labels).
        start_frame: Starting frame index.
        end_frame: Ending frame index (exclusive). None = all frames.
        step: Frame step size.

    Yields:
        FrameData objects for each frame.
    """
    if end_frame is None:
        end_frame = dataset.num_frames

    for frame_id in range(start_frame, end_frame, step):
        frame = dataset.get_frame(frame_id)

        # Run pipeline
        boxes, ground_mask, cluster_labels = pipeline_func(frame)

        yield FrameData(
            frame_id=frame_id,
            points=frame.points,
            boxes=boxes,
            ground_mask=ground_mask,
            cluster_labels=cluster_labels,
            title=f"Frame {frame_id:06d}",
        )
