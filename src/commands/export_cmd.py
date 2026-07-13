"""
Export command - Export results in various formats.
"""

from pathlib import Path

import typer

from ..datasets.kitti_odometry import KITTIDataset
from ..io.boxes_to_labels import BoxesToLabelsConverter, BoxLabelMapping
from ..io.export_predictions import PredictionExporter
from ..viz.renderer import CameraParams, RenderConfig
from ..viz.video_export import FrameData, VideoConfig, VideoExporter
from .common import load_config
from .viz_cmd import _process_single_frame_for_viz


def _export_predictions(
    cfg: dict, output_dir: Path, start: int, end: int | None
) -> None:
    """Export ground truth labels as predictions in SemanticKITTI format.

    This exports the ground truth semantic labels as predictions, which is useful for:
    - Testing the official SemanticKITTI evaluation pipeline
    - Validating the predictions directory structure
    - Creating baseline results

    Note: For actual detection-based predictions, the 'run' command would need to
    assign semantic labels to points based on detected boxes. Currently, this
    exports ground truth to verify the evaluation workflow.
    """
    dataset_root = Path(cfg["dataset"]["root"])
    seq = cfg["dataset"]["sequence"]

    typer.echo(f"Loading dataset: {dataset_root}, sequence: {seq}")
    dataset = KITTIDataset(dataset_root, seq)

    if not dataset.has_labels:
        raise typer.BadParameter(
            f"Sequence {seq} has no ground truth labels; cannot export predictions."
        )

    if end is None:
        end = dataset.num_frames

    typer.echo(f"Exporting ground truth as predictions for frames {start} to {end - 1}")
    typer.echo(f"Output directory: {output_dir}")

    # Initialize exporter
    exporter = PredictionExporter(
        predictions_root=output_dir,
        sequence=seq,
        dataset_root=dataset_root,
        semantic_ids_are_training=False,  # GT labels are in original format
    )

    def progress_callback(current: int, total: int) -> None:
        if current % 100 == 0 or current == total:
            typer.echo(f"  Exported {current}/{total} frames")

    stats = exporter.export_ground_truth_as_predictions(
        dataset=dataset,
        start_frame=start,
        end_frame=end,
        progress_callback=progress_callback,
    )

    typer.echo()
    if stats.num_failed:
        typer.echo("Export failed: one or more frames could not be exported.")
        typer.echo(f"  Frames exported: {stats.num_exported}/{stats.num_frames}")
        typer.echo(f"  Failed frames: {stats.num_failed}")
        typer.echo(f"  Failed IDs: {stats.failed_frames[:10]}...")
        raise typer.Exit(code=1)

    typer.echo("Export complete!")
    typer.echo(f"  Frames exported: {stats.num_exported}/{stats.num_frames}")
    typer.echo(f"  Total points: {stats.total_points:,}")
    typer.echo(f"  Output: {exporter.predictions_dir}")


def _export_predictions_from_boxes(
    cfg: dict, output_dir: Path, start: int, end: int | None
) -> None:
    """Export predictions generated from detection boxes.

    This converts detection boxes to per-point semantic labels, enabling
    evaluation of detection performance using the SemanticKITTI API.

    Points inside valid boxes are assigned a semantic class based on
    box size heuristics. Points outside all boxes are labeled as 'unlabeled'.
    """
    dataset_root = Path(cfg["dataset"]["root"])
    seq = cfg["dataset"]["sequence"]
    boxes_root = Path(cfg["output"]["root"]) / cfg["output"]["boxes_dir"]

    typer.echo(f"Loading dataset: {dataset_root}, sequence: {seq}")
    typer.echo(f"Boxes directory: {boxes_root}")

    # Check if boxes exist
    boxes_file = boxes_root / seq / "boxes.jsonl"
    if not boxes_file.exists():
        raise typer.BadParameter(
            f"No detection boxes found at {boxes_file}; run the detection pipeline first."
        )

    # Create converter with default mapping
    mapping = BoxLabelMapping(
        default_box_class=10,  # car
        background_class=0,  # unlabeled
        use_box_size_heuristics=True,
    )

    converter = BoxesToLabelsConverter(
        dataset_root=dataset_root,
        boxes_root=boxes_root,
        predictions_root=output_dir,
        sequence=seq,
        mapping=mapping,
    )

    # Get available frames
    available_frames = converter.get_available_frames()
    typer.echo(f"Found {len(available_frames)} frames with detection boxes")

    # Adjust range
    if end is None:
        end = converter.dataset.num_frames
    end = min(end, converter.dataset.num_frames)
    if start < 0:
        raise typer.BadParameter("export.start must be >= 0")
    if end <= start:
        raise typer.BadParameter("export.end must be > export.start")

    typer.echo(f"Converting boxes to labels for frames {start} to {end - 1}")
    typer.echo(f"Output directory: {output_dir}")
    converter.clear_output_dir()
    typer.echo(f"Cleared existing prediction labels in {converter.output_dir}")

    def progress_callback(current: int, total: int) -> None:
        if current % 100 == 0 or current == total:
            typer.echo(f"  Converted {current}/{total} frames")

    results = converter.convert_all_frames(
        start_frame=start,
        end_frame=end,
        progress_callback=progress_callback,
    )

    # Summary
    typer.echo()
    typer.echo("Conversion complete!")
    typer.echo(f"  Frames converted: {len(results)}")

    if results:
        total_points = sum(r.num_points for r in results)
        total_in_boxes = sum(r.points_in_boxes for r in results)
        coverage = total_in_boxes / total_points * 100 if total_points > 0 else 0
        typer.echo(f"  Total points: {total_points:,}")
        typer.echo(f"  Points in boxes: {total_in_boxes:,} ({coverage:.1f}%)")
        typer.echo(f"  Points outside boxes: {total_points - total_in_boxes:,}")

    typer.echo(f"  Output: {converter.output_dir}")


def _export_video(
    cfg: dict, output_dir: Path, start: int, end: int | None, step: int
) -> None:
    """Export video visualization."""
    try:
        dataset_root = Path(cfg["dataset"]["root"])
        seq = cfg["dataset"]["sequence"]
        dataset = KITTIDataset(dataset_root, seq)

        video_cfg = cfg.get("video", {}) or {}
        seq_cfg = (video_cfg.get("sequence", {}) or {}) if isinstance(video_cfg, dict) else {}
        max_frames = seq_cfg.get("max_frames", 500)

        try:
            fps = int(video_cfg.get("fps", 10))
        except (TypeError, ValueError):
            fps = 10

        try:
            width = int(video_cfg.get("width", 1280))
        except (TypeError, ValueError):
            width = 1280

        try:
            height = int(video_cfg.get("height", 720))
        except (TypeError, ValueError):
            height = 720

        if end is None:
            try:
                max_end = int(max_frames)
            except (TypeError, ValueError):
                max_end = 500
            end = min(max_end, dataset.num_frames)

        if step <= 0:
            step = 1

        if start < 0:
            start = 0
        start = min(start, dataset.num_frames)
        end = min(max(end, start + 1), dataset.num_frames)

        typer.echo(f"Processing frames {start} to {end - 1}, step {step}...")

        def frame_generator():
            for frame_idx in range(start, end, step):
                result = _process_single_frame_for_viz(cfg, dataset, frame_idx)
                yield FrameData(
                    frame_id=frame_idx,
                    points=result["points"],
                    boxes=result["boxes"],
                    ground_mask=result["ground_mask"],
                    cluster_labels=result["cluster_labels"],
                    title=f"Frame {frame_idx:06d} | {len(result['boxes'])} detections",
                )

        video_config = VideoConfig(fps=fps, width=width, height=height)
        point_size = (cfg.get("visualization", {}) or {}).get("point_size", 1.0)
        render_config = RenderConfig(width=width, height=height, point_size=point_size)
        exporter = VideoExporter(video_config, render_config)

        camera_cfg = cfg.get("camera", {}) or {}
        camera = CameraParams(
            elevation=camera_cfg.get("elevation", 60),
            azimuth=camera_cfg.get("azimuth", -90),
            distance=camera_cfg.get("distance", 50),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"sequence{seq}.mp4"

        def progress(current, total):
            if current % 20 == 0 or current == total:
                typer.echo(f"  Rendered {current}/{total} frames")

        frames_list = list(frame_generator())
        exporter.export_sequence(
            frames_list,
            output_file,
            camera=camera,
            mode="boxes",
            progress_callback=progress,
        )
        typer.echo(f"Saved video to {output_file}")

    except ImportError as exc:
        typer.echo(f"Import error: {exc}")
        typer.echo("Install video dependencies: uv pip install imageio imageio-ffmpeg")
        raise typer.Exit(code=1) from exc
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"Error during video export: {exc}")
        raise typer.Exit(code=1) from exc


def export_command(config: Path | None = None) -> None:
    """
    Export results in various formats.

    All parameters are configured via configs/export.yaml.
    Use --config to override with a custom config file.
    """
    cfg = load_config(config, task="export")

    # Get export settings from config
    export_cfg = cfg.get("export", {})
    output_type = export_cfg.get("type", "video")
    from_boxes = export_cfg.get("from_boxes", False)
    start_frame = export_cfg.get("start", 0)
    end_frame = export_cfg.get("end")
    frame_step = export_cfg.get("step", 1)

    # Get video settings
    video_cfg = cfg.get("video", {})
    seq_cfg = video_cfg.get("sequence", {})
    if frame_step == 1:
        frame_step = seq_cfg.get("step", 5)

    typer.echo("=" * 60)
    typer.echo("Export")
    typer.echo("=" * 60)
    typer.echo(f"Type: {output_type}")
    typer.echo(f"Sequence: {cfg['dataset']['sequence']}")

    # Determine output path
    default_paths = {
        "video": Path(cfg["output"]["root"]) / cfg["output"]["videos_dir"],
        "report": Path(cfg["output"]["root"]) / cfg["output"]["reports_dir"],
        "predictions": Path(cfg["output"]["root"]) / cfg["output"]["predictions_dir"],
    }
    output_path = default_paths.get(output_type, Path(cfg["output"]["root"]))

    typer.echo(f"Output: {output_path}")

    if output_type == "video":
        _export_video(cfg, output_path, start_frame, end_frame, frame_step)

    elif output_type == "report":
        typer.echo(
            "Reports are generated by the eval command (configured in configs/evaluation.yaml)."
        )
    elif output_type == "predictions":
        if from_boxes:
            _export_predictions_from_boxes(cfg, output_path, start_frame, end_frame)
        else:
            _export_predictions(cfg, output_path, start_frame, end_frame)
    else:
        raise typer.BadParameter(f"Unknown export type: {output_type}")
