#!/usr/bin/env python3
"""Generate visualization videos from detection results.

This is a convenience wrapper around the main CLI's export command.
It provides a standalone script interface for video generation.

Creates:
1. debug.mp4 - Single frame debug visualization (rotating view)
2. sequence{seq}.mp4 - Full sequence playback

Usage:
    uv run python tools/export_video.py
    uv run python tools/export_video.py --config configs/base.yaml
    uv run python tools/export_video.py --debug-only
    uv run python tools/export_video.py --sequence-only

Note: This script reuses the pipeline and export logic from src.commands.
For most use cases, prefer: uv run python -m src.main export (configured in configs/export.yaml)
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.commands.viz_cmd import _process_single_frame_for_viz
from src.utils.config import load_config


def export_debug_video(
    config: dict,
    dataset,
    frame_idx: int,
    output_path: Path,
) -> None:
    """Export debug video for a single frame (rotating view)."""
    from src.viz.renderer import RenderConfig
    from src.viz.video_export import FrameData, VideoConfig, VideoExporter

    print(f"Processing frame {frame_idx} for debug video...")
    result = _process_single_frame_for_viz(config, dataset, frame_idx)

    frame_data = FrameData(
        frame_id=frame_idx,
        points=result["points"],
        boxes=result["boxes"],
        ground_mask=result["ground_mask"],
        cluster_labels=result["cluster_labels"],
    )

    # Get video settings from config
    video_cfg = config.get("video", {})
    debug_cfg = video_cfg.get("debug", {})

    video_config = VideoConfig(
        fps=video_cfg.get("fps", 15),
        width=video_cfg.get("width", 1280),
        height=video_cfg.get("height", 720),
    )
    render_config = RenderConfig(
        width=video_cfg.get("width", 1280),
        height=video_cfg.get("height", 720),
    )
    exporter = VideoExporter(video_config, render_config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exporter.export_debug_video(
        frame_data,
        output_path,
        num_views=debug_cfg.get("num_views", 4),
        frames_per_view=debug_cfg.get("frames_per_view", 45),
    )
    print(f"Debug video saved to {output_path}")


def export_sequence_video(
    config: dict,
    dataset,
    output_path: Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    step: int = 1,
) -> None:
    """Export full sequence video."""
    from src.viz.renderer import CameraParams, RenderConfig
    from src.viz.video_export import FrameData, VideoConfig, VideoExporter

    if end_frame is None:
        end_frame = dataset.num_frames

    print(f"Processing sequence, frames {start_frame}-{end_frame}, step {step}...")

    def frame_generator():
        for frame_idx in range(start_frame, end_frame, step):
            try:
                result = _process_single_frame_for_viz(config, dataset, frame_idx)
                yield FrameData(
                    frame_id=frame_idx,
                    points=result["points"],
                    boxes=result["boxes"],
                    ground_mask=result["ground_mask"],
                    cluster_labels=result["cluster_labels"],
                    title=f"Frame {frame_idx:06d} | {len(result['boxes'])} detections",
                )
            except Exception as e:
                print(f"  Error processing frame {frame_idx}: {e}")
                continue

    # Get video settings from config
    video_cfg = config.get("video", {})

    video_config = VideoConfig(
        fps=video_cfg.get("fps", 10),
        width=video_cfg.get("width", 1280),
        height=video_cfg.get("height", 720),
    )
    render_config = RenderConfig(
        width=video_cfg.get("width", 1280),
        height=video_cfg.get("height", 720),
    )
    exporter = VideoExporter(video_config, render_config)

    # Get camera settings from config
    camera_cfg = config.get("camera", {})
    camera = CameraParams(
        elevation=camera_cfg.get("elevation", 60),
        azimuth=camera_cfg.get("azimuth", -90),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def progress(current, total):
        if current % 20 == 0 or current == total:
            print(f"  Rendered {current}/{total} frames")

    exporter.export_sequence(
        list(frame_generator()),
        output_path,
        camera=camera,
        mode="boxes",
        progress_callback=progress,
    )
    print(f"Sequence video saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export visualization videos",
        epilog="For more options, use: uv run python -m src.main export --help",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (default: use configs/*.yaml)",
    )
    parser.add_argument(
        "--debug-only",
        action="store_true",
        help="Only export debug video (single frame rotating view)",
    )
    parser.add_argument(
        "--sequence-only",
        action="store_true",
        help="Only export sequence video",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Video Export")
    print("=" * 60)

    # Load config using task-based loading
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config(task="export")

    # Get settings from config
    export_cfg = config.get("export", {})
    video_cfg = config.get("video", {})
    debug_cfg = video_cfg.get("debug", {})
    seq_cfg = video_cfg.get("sequence", {})

    sequence = config["dataset"]["sequence"]
    frame = debug_cfg.get("frame", 100)
    start_frame = export_cfg.get("start", 0)
    end_frame = export_cfg.get("end")
    step = seq_cfg.get("step", 5)
    output_dir = Path(config["output"]["root"]) / config["output"]["videos_dir"]

    # Import dataset
    from src.datasets.kitti_odometry import KITTIDataset

    dataset_root = Path(config["dataset"]["root"])
    print(f"Loading dataset from {dataset_root}, sequence {sequence}")
    dataset = KITTIDataset(dataset_root, sequence)
    print(f"Dataset has {dataset.num_frames} frames")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export debug video
    if not args.sequence_only:
        debug_path = output_dir / "debug.mp4"
        print("\n--- Exporting debug video ---")
        export_debug_video(config, dataset, frame, debug_path)

    # Export sequence video
    if not args.debug_only:
        seq_path = output_dir / f"sequence{sequence}.mp4"
        print("\n--- Exporting sequence video ---")
        end = end_frame if end_frame is not None else min(500, dataset.num_frames)
        export_sequence_video(
            config,
            dataset,
            seq_path,
            start_frame=start_frame,
            end_frame=end,
            step=step,
        )

    print("\n" + "=" * 60)
    print("Video export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
