"""
Viz command - Visualize point clouds and detection results.
"""

from pathlib import Path
from typing import Any

import numpy as np
import typer

from ..boxes.obb import fit_obbs_to_clusters
from ..datasets.kitti_odometry import KITTIDataset
from ..preprocess.ground import segment_ground
from ..preprocess.roi import ROIBounds, crop_points
from ..preprocess.voxel import downsample_points
from ..proposals.bev_cc import cluster_bev_cc
from ..proposals.dbscan import cluster_dbscan_from_config
from ..proposals.filtering import filter_clusters
from ..viz.open3d_viewer import view_frame_open3d
from ..viz.renderer import CameraParams, RenderConfig
from ..viz.video_export import FrameData, VideoConfig, VideoExporter
from .common import load_config


def _process_single_frame_for_viz(
    config: dict[str, Any], dataset, frame_idx: int
) -> dict[str, object]:
    """Process a single frame through the pipeline for visualization."""
    frame = dataset.get_frame(frame_idx)
    points = frame.points.copy()

    # ROI crop
    roi_cfg = config.get("preprocess", {}).get("roi", {})
    roi_bounds = ROIBounds(
        x_min=roi_cfg.get("x_min", 0),
        x_max=roi_cfg.get("x_max", 70),
        y_min=roi_cfg.get("y_min", -40),
        y_max=roi_cfg.get("y_max", 40),
        z_min=roi_cfg.get("z_min", -3),
        z_max=roi_cfg.get("z_max", 3),
    )
    roi_result = crop_points(points, roi_bounds)
    points = roi_result.points

    # Voxel downsample
    voxel_cfg = config.get("preprocess", {}).get("voxel", {})
    if voxel_cfg.get("enabled", True):
        voxel_result = downsample_points(points, voxel_cfg)
        points = voxel_result.points

    # Ground segmentation
    ground_cfg = config.get("ground", {})
    ground_result = segment_ground(
        points,
        distance_threshold=ground_cfg.get("distance_threshold", 0.2),
        ransac_n=ground_cfg.get("ransac_n", 3),
        num_iterations=ground_cfg.get("num_iterations", 1000),
        z_min=ground_cfg.get("z_min", -2.5),
        z_max=ground_cfg.get("z_max", -0.5),
        min_normal_z=ground_cfg.get("min_normal_z", 0.9),
        random_seed=ground_cfg.get("random_seed", 0),
    )
    non_ground_points = ground_result.nonground_points

    # Proposal generation
    proposal_method = config.get("proposals", {}).get("method", "bev_cc")
    if proposal_method == "bev_cc":
        bev_cfg = config.get("proposals", {}).get("bev_cc", {})
        proposal_result = cluster_bev_cc(
            non_ground_points,
            resolution=bev_cfg.get("resolution", 0.2),
            min_height=bev_cfg.get("min_height", 0.3),
            min_points=bev_cfg.get("min_points", 10),
        )
    elif proposal_method == "dbscan":
        dbscan_cfg = config.get("proposals", {}).get("dbscan", {})
        proposal_result = cluster_dbscan_from_config(non_ground_points, dbscan_cfg)
    else:
        raise typer.BadParameter(f"Unsupported proposal method: {proposal_method}")
    clusters = proposal_result.clusters
    cluster_labels = proposal_result.labels

    # Filter clusters
    filter_cfg = config.get("filtering", {})
    filter_result = filter_clusters(non_ground_points, clusters, filter_cfg)
    filtered_clusters = filter_result.clusters

    # Fit boxes
    box_cfg = config.get("boxes", {})
    box_method = box_cfg.get("method", "obb")
    if box_method != "obb":
        raise typer.BadParameter(
            f"Unsupported boxes.method: {box_method}. Only 'obb' is supported."
        )
    box_results = fit_obbs_to_clusters(
        non_ground_points,
        filtered_clusters,
        z_percentile=box_cfg.get("z_percentile", 5.0),
        min_volume=box_cfg.get("min_box_volume", 0.1),
        max_volume=box_cfg.get("max_box_volume", 1000),
    )

    boxes = [br.params for br in box_results if br.valid and br.params is not None]

    # Create full cluster labels array for all points (ground points get -1)
    full_cluster_labels = np.full(len(points), -1, dtype=np.int32)
    full_cluster_labels[~ground_result.ground_mask] = cluster_labels

    return {
        "points": points,
        "ground_mask": ground_result.ground_mask,
        "non_ground_points": non_ground_points,
        "cluster_labels": full_cluster_labels,
        "boxes": boxes,
    }


def viz_command(
    config: Path | None = None,
    frame: int | None = None,
    mode: str | None = None,
    output: Path | None = None,
) -> None:
    """
    Visualize point clouds and detection results.

    All parameters are configured via configs/visualization.yaml.
    Use --config to override with a custom config file.
    """
    cfg = load_config(config, task="viz")

    # Get visualization settings from config
    viz_cfg = cfg.get("visualization", {})
    if mode is not None:
        viz_cfg["mode"] = mode
    if frame is not None:
        viz_cfg["frame"] = frame

    mode = viz_cfg.get("mode", "boxes")
    frame = viz_cfg.get("frame", 0)
    predictions = viz_cfg.get("show_boxes", True)
    max_points = viz_cfg.get("max_points", 0)
    point_size = viz_cfg.get("point_size", 2.0)

    typer.echo("=" * 60)
    typer.echo("Visualization")
    typer.echo("=" * 60)
    typer.echo(f"Sequence: {cfg['dataset']['sequence']}")
    typer.echo(f"Frame: {frame}")
    typer.echo(f"Mode: {mode}")
    typer.echo(f"Show predictions: {predictions}")

    # Run visualization
    try:
        dataset_root = Path(cfg["dataset"]["root"])
        seq = cfg["dataset"]["sequence"]
        dataset = KITTIDataset(dataset_root, seq)

        if frame >= dataset.num_frames:
            raise typer.BadParameter(
                f"Frame {frame} out of range (0-{dataset.num_frames - 1})"
            )

        typer.echo(f"\nProcessing frame {frame}...")
        result = _process_single_frame_for_viz(cfg, dataset, frame)

        # Create frame data for visualization
        frame_data = FrameData(
            frame_id=frame,
            points=result["points"],
            boxes=result["boxes"],
            ground_mask=result["ground_mask"],
            cluster_labels=result["cluster_labels"],
        )

        # Interactive viewer (default)
        if output is None:
            max_pts = None if max_points <= 0 else int(max_points)
            show_preds = predictions or mode in ("debug", "boxes")
            view_frame_open3d(
                frame_data,
                mode=mode,
                show_predictions=show_preds,
                max_points=max_pts,
                point_size=float(point_size),
            )
            return

        # Setup exporter (uses renderer internally)
        video_config = VideoConfig(fps=10, width=1280, height=720)
        render_config = RenderConfig(width=1280, height=720)
        exporter = VideoExporter(video_config, render_config)
        camera = CameraParams(elevation=30, azimuth=-60)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        render_mode = mode
        if render_mode == "debug":
            render_mode = "boxes" if (predictions or frame_data.boxes) else "points"

        # Use the internal render method
        image = exporter._render_frame(frame_data, camera, render_mode)

        try:
            import imageio

            imageio.imwrite(str(output_path), image)
            typer.echo(f"Saved visualization to {output_path}")
        except ImportError as exc:
            typer.echo("Error: imageio is required to save an image.")
            typer.echo("Install with: uv pip install imageio")
            raise typer.Exit(code=1) from exc

    except typer.Exit:
        raise
    except typer.BadParameter:
        raise
    except Exception as exc:
        typer.echo(f"Error during visualization: {exc}")
        raise typer.Exit(code=1) from exc
