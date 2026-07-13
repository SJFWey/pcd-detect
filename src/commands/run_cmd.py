"""
Run command - Execute the full detection pipeline.
"""

from pathlib import Path
from typing import Any

import typer

from ..boxes.obb import fit_obbs_to_clusters
from ..datasets.kitti_odometry import KITTIDataset
from ..io.export_boxes import BoxesExporter, create_frame_boxes
from ..preprocess.ground import segment_ground
from ..preprocess.roi import ROIBounds, crop_points
from ..preprocess.voxel import (
    downsample_points,
    voxel_downsample_fixed,
)
from ..proposals.bev_cc import cluster_bev_cc
from ..proposals.dbscan import cluster_dbscan_from_config
from ..proposals.filtering import filter_clusters
from ..utils.timer import get_timing_manager
from .common import load_config, parse_frame_range, print_config


def _run_detection_pipeline(cfg: dict[str, Any], frames_str: str | None) -> None:
    """Run the full detection pipeline on a sequence."""
    timer_manager = get_timing_manager()
    timer_manager.reset()

    # Load dataset
    dataset_root = Path(cfg["dataset"]["root"])
    sequence = cfg["dataset"]["sequence"]

    typer.echo("=" * 60)
    typer.echo(f"Running Detection Pipeline - Sequence {sequence}")
    typer.echo("=" * 60)

    dataset = KITTIDataset(dataset_root, sequence)
    typer.echo(f"Loaded dataset: {dataset.num_frames} frames")

    # Resolve frame range
    if dataset.num_frames <= 0:
        raise typer.BadParameter("Dataset contains no frames")

    frames_cfg = cfg.get("frames", {})
    step_raw = frames_cfg.get("step", 1)

    if step_raw is None:
        step_raw = 1

    try:
        step = int(step_raw)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter("frames.step must be an integer") from exc

    if step <= 0:
        raise typer.BadParameter("frames.step must be >= 1")

    if frames_str is not None:
        base_range = parse_frame_range(frames_str, dataset.num_frames)
        if base_range.start < 0:
            raise typer.BadParameter("frames must start at >= 0")
        if base_range.stop <= base_range.start:
            raise typer.BadParameter("frames selects no frames")
        start_frame = base_range.start
        end_frame = base_range.stop
    else:
        start_raw = frames_cfg.get("start", 0)
        end_raw = frames_cfg.get("end")

        if start_raw is None:
            start_raw = 0

        try:
            start_frame = int(start_raw)
        except (TypeError, ValueError) as exc:
            raise typer.BadParameter("frames.start must be an integer") from exc

        if start_frame < 0:
            raise typer.BadParameter("frames.start must be >= 0")
        if start_frame >= dataset.num_frames:
            raise typer.BadParameter(
                f"frames.start ({start_frame}) out of range (0-{dataset.num_frames - 1})"
            )

        if end_raw is None:
            end_frame = dataset.num_frames
        else:
            try:
                end_frame = int(end_raw)
            except (TypeError, ValueError) as exc:
                raise typer.BadParameter("frames.end must be an integer or null") from exc
            if end_frame < 0:
                raise typer.BadParameter("frames.end must be >= 0")
            end_frame = min(end_frame, dataset.num_frames)

    if end_frame <= start_frame:
        raise typer.BadParameter("frames.end must be > frames.start")

    frame_range = range(start_frame, end_frame, step)
    if step == 1:
        typer.echo(f"Processing frames: {start_frame} to {end_frame - 1}")
    else:
        typer.echo(f"Processing frames: {start_frame} to {end_frame - 1} (step {step})")

    # Setup output directories
    output_root = Path(cfg["output"]["root"])
    boxes_dir = cfg["output"].get("boxes_dir", "boxes")
    boxes_format = cfg["output"].get("boxes_format", "jsonl")

    # Initialize exporter
    boxes_exporter = BoxesExporter(
        output_root,
        sequence,
        boxes_dir=boxes_dir,
        format=boxes_format,
    )

    # Get config sections
    preprocess_cfg = cfg.get("preprocess", {})
    roi_cfg = preprocess_cfg.get("roi", {})
    voxel_cfg = preprocess_cfg.get("voxel", {})
    ground_cfg = cfg.get("ground", {})
    proposal_cfg = cfg.get("proposals", {})
    proposal_method = proposal_cfg.get("method", "bev_cc")
    bev_cfg = proposal_cfg.get("bev_cc", {})
    dbscan_cfg = proposal_cfg.get("dbscan", {})
    filter_cfg = cfg.get("filtering", {})
    box_cfg = cfg.get("boxes", {})
    box_method = box_cfg.get("method", "obb")
    if box_method != "obb":
        raise typer.BadParameter(
            f"Unsupported boxes.method: {box_method}. Only 'obb' is supported."
        )

    # Build ROI bounds
    roi_bounds = ROIBounds(
        x_min=roi_cfg.get("x_min", 0),
        x_max=roi_cfg.get("x_max", 70),
        y_min=roi_cfg.get("y_min", -40),
        y_max=roi_cfg.get("y_max", 40),
        z_min=roi_cfg.get("z_min", -3),
        z_max=roi_cfg.get("z_max", 3),
    )

    total_boxes = 0

    for frame_idx in frame_range:
        # Load frame
        with timer_manager.measure("load", frame_idx):
            frame = dataset.get_frame(frame_idx)
            points = frame.points.copy()

        input_points = points.shape[0]

        # ROI crop
        with timer_manager.measure("roi_crop", frame_idx, input_points) as t:
            roi_result = crop_points(points, roi_bounds)
            points = roi_result.points
            t.set_output_points(roi_result.output_points)

        # Voxel downsampling
        if voxel_cfg.get("enabled", True):
            with timer_manager.measure("voxel", frame_idx, points.shape[0]) as t:
                voxel_mode = voxel_cfg.get("mode", "fixed")
                if voxel_mode == "distance_adaptive":
                    voxel_result = downsample_points(points, voxel_cfg)
                elif voxel_mode == "fixed":
                    voxel_result = voxel_downsample_fixed(
                        points, voxel_cfg.get("voxel_size", 0.1)
                    )
                else:
                    raise typer.BadParameter(f"Unsupported voxel mode: {voxel_mode}")
                points = voxel_result.points
                t.set_output_points(voxel_result.output_points)

        # Ground segmentation
        with timer_manager.measure("ground", frame_idx, points.shape[0]) as t:
            ground_result = segment_ground(
                points,
                distance_threshold=ground_cfg.get("distance_threshold", 0.2),
                ransac_n=ground_cfg.get("ransac_n", 3),
                num_iterations=ground_cfg.get("num_iterations", 1000),
                z_min=ground_cfg.get("z_min"),
                z_max=ground_cfg.get("z_max"),
                min_normal_z=ground_cfg.get("min_normal_z", 0.9),
                random_seed=ground_cfg.get("random_seed", 0),
            )
            non_ground_points = ground_result.nonground_points
            t.set_output_points(non_ground_points.shape[0])

        # Proposal generation
        with timer_manager.measure(
            "proposals", frame_idx, non_ground_points.shape[0]
        ) as t:
            if proposal_method == "bev_cc":
                proposal_result = cluster_bev_cc(
                    non_ground_points,
                    resolution=bev_cfg.get("resolution", 0.2),
                    min_height=bev_cfg.get("min_height", 0.3),
                    min_points=bev_cfg.get("min_points", 10),
                )
            elif proposal_method == "dbscan":
                proposal_result = cluster_dbscan_from_config(non_ground_points, dbscan_cfg)
            else:
                raise typer.BadParameter(
                    f"Unsupported proposal method: {proposal_method}"
                )
            clusters = proposal_result.clusters
            t.set_output_points(proposal_result.clustered_points)

        # Filter clusters
        with timer_manager.measure("filtering", frame_idx):
            filter_result = filter_clusters(non_ground_points, clusters, filter_cfg)
            filtered_clusters = filter_result.clusters

        # Fit bounding boxes
        with timer_manager.measure("boxes", frame_idx):
            box_results = fit_obbs_to_clusters(
                non_ground_points,
                filtered_clusters,
                z_percentile=box_cfg.get("z_percentile", 5.0),
                min_volume=box_cfg.get("min_box_volume", 0.1),
                max_volume=box_cfg.get("max_box_volume", 1000),
            )

        # Export boxes
        with timer_manager.measure("export", frame_idx):
            frame_boxes = create_frame_boxes(frame_idx, box_results)
            boxes_exporter.export_frame(frame_boxes)

        valid_boxes = sum(1 for br in box_results if br.valid)
        total_boxes += valid_boxes

        # Progress logging
        if (frame_idx + 1) % 50 == 0 or frame_idx == frame_range.start:
            typer.echo(
                f"  Frame {frame_idx:4d}: {input_points:6d} pts -> "
                f"{roi_result.output_points:5d} (ROI) -> "
                f"{non_ground_points.shape[0]:5d} (non-ground) -> "
                f"{len(filtered_clusters):3d} clusters -> "
                f"{valid_boxes:3d} boxes"
            )

    # Finalize exports
    boxes_path = boxes_exporter.finalize()

    typer.echo()
    typer.echo("-" * 60)
    typer.echo("Pipeline Complete!")
    typer.echo("-" * 60)
    typer.echo(f"  Frames processed: {len(frame_range)}")
    typer.echo(f"  Total boxes: {total_boxes}")
    typer.echo(f"  Boxes file: {boxes_path}")

    # Print timing summary
    typer.echo()
    typer.echo(timer_manager.summary())


def run_command(
    config: Path | None = None,
    sequence: str | None = None,
    frames: str | None = None,
    dry_run: bool = False,
) -> None:
    """
    Run the full detection pipeline.

    All parameters are configured via configs/base.yaml and configs/detection.yaml.
    Use --config to override with a custom config file.
    """
    cfg = load_config(config, task="run")

    if sequence is not None:
        cfg.setdefault("dataset", {})["sequence"] = sequence

    if dry_run:
        typer.echo("=" * 60)
        typer.echo("DRY RUN - Execution Plan")
        typer.echo("=" * 60)
        typer.echo("\n[Configuration]")
        print_config(cfg)
        if frames is not None:
            typer.echo("\n[CLI Overrides]")
            typer.echo(f"frames: {frames}")
        typer.echo("\n[Pipeline Steps]")

        proposal_method = cfg.get("proposals", {}).get("method", "bev_cc")
        steps = [
            "1. Load KITTI Odometry point clouds",
            "2. Load SemanticKITTI labels",
            "3. Apply ROI crop",
            "4. Voxel downsampling",
            "5. Ground segmentation (RANSAC)",
            f"6. Proposal generation ({proposal_method})",
            "7. Cluster filtering",
            "8. OBB fitting",
            "9. Export boxes",
        ]
        for step in steps:
            typer.echo(f"  {step}")

        typer.echo("\n[Output]")
        output_cfg = cfg.get("output", {})
        dataset_cfg = cfg.get("dataset", {})
        output_root = Path(output_cfg.get("root", "outputs"))
        sequence = dataset_cfg.get("sequence", "")
        boxes_dir = output_cfg.get("boxes_dir", "boxes")
        typer.echo(f"  Boxes: {output_root}/{boxes_dir}/{sequence}")
        typer.echo(f"  Predictions: {output_root}/{output_cfg.get('predictions_dir')}")
        typer.echo(f"  Reports: {output_root}/{output_cfg.get('reports_dir')}")
        return

    # Run the detection pipeline
    _run_detection_pipeline(cfg, frames)
