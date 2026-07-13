#!/usr/bin/env python3
"""Render a static bird's-eye-view image for qualitative inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.commands.common import load_config
from src.commands.eval_cmd import (
    _ids_match_target,
    _resolve_detection_target_ids,
    _semantic_id_from_box_record,
)
from src.datasets.semkitti_labels import SemanticKITTILabels
from src.eval.detection_eval import BoundingBox3D, InstanceMatcher, extract_gt_instances
from src.io.boxes_to_labels import BoxLabelMapping
from src.preprocess.roi import ROIBounds, apply_mask, roi_mask
from src.commands.viz_cmd import _process_single_frame_for_viz
from src.datasets.kitti_odometry import KITTIDataset


FIG_BG = "#ffffff"
AX_BG = "#ffffff"
GRID = "#d8d8d8"
TEXT = "#222222"
SPINE = "#777777"
GROUND = "#9bbbd4"
NOISE = "#d0d0d0"
BOX = "#d62728"


def _display_bounds(config: dict) -> ROIBounds:
    """Resolve the XY window used for the rendered qualitative view."""
    roi_cfg = config.get("preprocess", {}).get("roi", {})
    display_cfg = config.get("visualization", {}).get("display_roi") or roi_cfg
    return ROIBounds(
        x_min=float(display_cfg.get("x_min", roi_cfg.get("x_min", 0.0))),
        x_max=float(display_cfg.get("x_max", roi_cfg.get("x_max", 70.0))),
        y_min=float(display_cfg.get("y_min", roi_cfg.get("y_min", -40.0))),
        y_max=float(display_cfg.get("y_max", roi_cfg.get("y_max", 40.0))),
        z_min=float(roi_cfg.get("z_min", -3.0)),
        z_max=float(roi_cfg.get("z_max", 3.0)),
    )


def _boxes_in_display_roi(boxes, bounds: ROIBounds):
    """Keep boxes whose centre is inside the rendered view."""
    return [
        box
        for box in boxes
        if bounds.x_min <= box.center[0] <= bounds.x_max
        and bounds.y_min <= box.center[1] <= bounds.y_max
    ]


def _detection_overlay_data(config: dict, dataset: KITTIDataset, frame_idx: int, boxes) -> tuple[list[BoundingBox3D], list[BoundingBox3D], set[int], set[int]]:
    """Build a label-backed overlay using the same target and IoU rules as eval."""
    frame = dataset.get_frame(frame_idx)
    if frame.inst_label is None or frame.sem_label is None:
        raise RuntimeError("Detection overlay requires SemanticKITTI instance labels")

    labels_helper = SemanticKITTILabels()
    _, target_ids, target_name_to_ids = _resolve_detection_target_ids(
        config, labels_helper
    )
    roi_cfg = config.get("preprocess", {}).get("roi", {})
    bounds = ROIBounds(
        x_min=float(roi_cfg.get("x_min", 0.0)),
        x_max=float(roi_cfg.get("x_max", 70.0)),
        y_min=float(roi_cfg.get("y_min", -40.0)),
        y_max=float(roi_cfg.get("y_max", 40.0)),
        z_min=float(roi_cfg.get("z_min", -3.0)),
        z_max=float(roi_cfg.get("z_max", 3.0)),
    )
    points, instance_labels, semantic_labels = apply_mask(
        roi_mask(frame.points, bounds),
        frame.points,
        frame.inst_label,
        frame.sem_label,
    )
    ground_truth = [
        box
        for box in extract_gt_instances(
            points, instance_labels, semantic_labels, labels_helper
        )
        if _ids_match_target(box.semantic_id, target_ids)
    ]

    class_mapping = BoxLabelMapping()
    predictions = []
    for box in boxes:
        semantic_id, _ = _semantic_id_from_box_record(
            box.to_dict(), class_mapping, target_name_to_ids
        )
        if _ids_match_target(semantic_id, target_ids):
            predictions.append(BoundingBox3D.from_obb_params(box, semantic_id=semantic_id))

    display_bounds = _display_bounds(config)
    predictions = _boxes_in_display_roi(predictions, display_bounds)
    ground_truth = _boxes_in_display_roi(ground_truth, display_bounds)

    threshold = float(config.get("evaluation", {}).get("iou_threshold", 0.5))
    matches, _, _ = InstanceMatcher(
        iou_threshold=threshold, use_3d_iou=False, match_classes=True
    ).match(predictions, ground_truth)
    return (
        predictions,
        ground_truth,
        {pred_idx for pred_idx, _, _ in matches},
        {gt_idx for _, gt_idx, _ in matches},
    )


def _box_bottom_xy(box) -> np.ndarray:
    corners = np.asarray(box.corners, dtype=np.float64)
    return corners[:4, :2]


def _style_axis(ax, config: dict, points: np.ndarray, title: str) -> None:
    roi_cfg = config.get("preprocess", {}).get("roi", {})
    display_roi = config.get("visualization", {}).get("display_roi") or roi_cfg
    ax.set_xlim(
        float(display_roi.get("x_min", points[:, 0].min())),
        float(display_roi.get("x_max", points[:, 0].max())),
    )
    ax.set_ylim(
        float(display_roi.get("y_min", points[:, 1].min())),
        float(display_roi.get("y_max", points[:, 1].max())),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=GRID, linewidth=0.5)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_xlabel("x forward (m)", color=TEXT)
    ax.set_ylabel("y left (m)", color=TEXT)
    ax.set_title(title, color=TEXT, pad=10, fontsize=11)
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_color(SPINE)


def _draw_ground(ax, points: np.ndarray, ground_mask: np.ndarray | None) -> None:
    if ground_mask is not None and np.any(ground_mask):
        ax.scatter(
            points[ground_mask, 0],
            points[ground_mask, 1],
            s=0.35,
            c=GROUND,
            alpha=0.45,
            linewidths=0,
        )
    nonground_mask = (
        ~ground_mask if ground_mask is not None else np.ones(points.shape[0], dtype=bool)
    )
    if np.any(nonground_mask):
        z = points[nonground_mask, 2]
        ax.scatter(
            points[nonground_mask, 0],
            points[nonground_mask, 1],
            s=1.0,
            c=z,
            cmap="plasma",
            alpha=0.85,
            linewidths=0,
        )


def _draw_clusters(ax, points: np.ndarray, cluster_labels: np.ndarray | None) -> None:
    if cluster_labels is None:
        ax.scatter(points[:, 0], points[:, 1], s=0.4, c=NOISE, linewidths=0)
        return

    labels = np.asarray(cluster_labels)
    noise_mask = labels < 0
    if np.any(noise_mask):
        ax.scatter(
            points[noise_mask, 0],
            points[noise_mask, 1],
            s=0.35,
            c=NOISE,
            alpha=0.35,
            linewidths=0,
        )
    valid_mask = labels >= 0
    if np.any(valid_mask):
        ax.scatter(
            points[valid_mask, 0],
            points[valid_mask, 1],
            s=1.1,
            c=labels[valid_mask],
                cmap="tab20",
                alpha=0.88,
                linewidths=0,
            )


def _draw_boxes(ax, boxes) -> None:
    from matplotlib.patches import Polygon

    for box in boxes:
        patch = Polygon(
            _box_bottom_xy(box),
            closed=True,
            fill=False,
            edgecolor=BOX,
            linewidth=1.2,
            alpha=0.95,
        )
        ax.add_patch(patch)


def _render_bev(config: dict, frame_idx: int, mode: str, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = KITTIDataset(config["dataset"]["root"], config["dataset"]["sequence"])
    result = _process_single_frame_for_viz(config, dataset, frame_idx)

    points = result["points"]
    boxes = result["boxes"]
    cluster_labels = result["cluster_labels"]
    ground_mask = result["ground_mask"]

    if mode == "pipeline":
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150, sharex=True, sharey=True)
        fig.patch.set_facecolor(FIG_BG)

        _draw_ground(axes[0], points, ground_mask)
        _style_axis(axes[0], config, points, "1. Ground vs. non-ground")
        # Height colorbar for plasma colormap on non-ground points
        from matplotlib.colors import Normalize
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=Normalize(vmin=float(points[:, 2].min()), vmax=float(points[:, 2].max())))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[0], fraction=0.046, pad=0.04, label="height (m)")

        _draw_clusters(axes[1], points, cluster_labels)
        _style_axis(axes[1], config, points, "2. DBSCAN clusters\n(color = cluster ID, not class)")

        # Panel 3: detection overlay (GT vs predictions)
        from src.viz.detection_overlay import (
            draw_detection_overlay,
            draw_distance_scale,
            draw_metrics_text,
        )

        predictions, ground_truth, matched_predictions, matched_ground_truth = (
            _detection_overlay_data(config, dataset, frame_idx, boxes)
        )
        _draw_ground(axes[2], points, ground_mask)
        counts = draw_detection_overlay(
            axes[2],
            predictions=predictions,
            ground_truth=ground_truth,
            matched_pred_indices=matched_predictions,
            matched_gt_indices=matched_ground_truth,
            show_gt=True,
            show_legend=True,
        )
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        display_bounds = _display_bounds(config)
        draw_metrics_text(
            axes[2],
            tp,
            fp,
            fn,
            precision,
            recall,
            f1,
            frame_idx,
            distance_str=(
                f"visible ROI: x={display_bounds.x_min:.0f}-{display_bounds.x_max:.0f}m"
            ),
        )
        _style_axis(
            axes[2], config, points, "3. GT-matched overlay (visible ROI)"
        )
        roi_cfg = config.get("preprocess", {}).get("roi", {})
        display_roi = config.get("visualization", {}).get("display_roi") or roi_cfg
        draw_distance_scale(
            axes[2],
            roi_x_range=(
                float(display_roi.get("x_min", points[:, 0].min())),
                float(display_roi.get("x_max", points[:, 0].max())),
            ),
        )

        fig.suptitle(
            f"SemanticKITTI 08 frame {frame_idx:06d} - BEV pipeline",
            color=TEXT,
            fontsize=14,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    if mode == "clusters" and cluster_labels is not None:
        _draw_clusters(ax, points, cluster_labels)
    else:
        _draw_ground(ax, points, ground_mask)
        _draw_boxes(ax, boxes)

    _style_axis(ax, config, points, f"SemanticKITTI 08 frame {frame_idx:06d} - BEV {mode}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a BEV qualitative image")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=["boxes", "clusters", "pipeline"],
        default="boxes",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config, task="viz")
    _render_bev(config, args.frame, args.mode, args.output)
    print(f"Saved BEV visualization to {args.output}")


if __name__ == "__main__":
    main()
