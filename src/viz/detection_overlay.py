"""Detection overlay visualization for BEV plots.

Renders TP / FP / FN / GT boxes on a matplotlib Axes using a
colorblind-safe palette designed for GitHub-light backgrounds.
Pure matplotlib — no Open3D dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# ── Colour palette (colorblind-safe, light-bg readable) ─────────────────
COLOR_TP: tuple[float, ...] = (0.173, 0.627, 0.173, 1.0)   # #2ca02c green
COLOR_FP: tuple[float, ...] = (0.839, 0.153, 0.157, 1.0)   # #d62728 red
COLOR_FN: tuple[float, ...] = (1.000, 0.498, 0.055, 1.0)    # #ff7f0e orange
COLOR_GT: tuple[float, ...] = (0.122, 0.467, 0.706, 1.0)    # #1f77b4 blue

_FILL_ALPHA = 0.15

# ── Helpers ──────────────────────────────────────────────────────────────

def _bottom_xy(box: Any) -> np.ndarray:
    """Return a BEV polygon from either a proposal or evaluation box."""
    if hasattr(box, "corners"):
        corners = np.asarray(box.corners, dtype=np.float64)
        return corners[:4, :2]

    length, width = np.asarray(box.dimensions, dtype=np.float64)[:2]
    half_length, half_width = length / 2.0, width / 2.0
    local = np.array(
        [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )
    cos_yaw, sin_yaw = np.cos(box.yaw), np.sin(box.yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    return (rotation @ local.T).T + np.asarray(box.center, dtype=np.float64)[:2]


def _add_box_patch(
    ax: Axes,
    box: Any,
    edgecolor: tuple[float, ...],
    linestyle: str,
    linewidth: float = 1.2,
    fill: bool = False,
    fill_alpha: float = _FILL_ALPHA,
    label: str | None = None,
) -> None:
    from matplotlib.patches import Polygon

    patch = Polygon(
        _bottom_xy(box),
        closed=True,
        fill=False,
        edgecolor=edgecolor,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )
    if fill:
        # Embed alpha in face RGBA so edge stays fully opaque
        patch.set_facecolor((*edgecolor[:3], fill_alpha))
        patch.set_edgecolor(edgecolor)
    ax.add_patch(patch)


# ── Public API ───────────────────────────────────────────────────────────

def draw_detection_overlay(
    ax: Axes,
    predictions: Sequence[Any],
    ground_truth: Sequence[Any],
    matched_pred_indices: set[int],
    matched_gt_indices: set[int],
    *,
    show_gt: bool = True,
    show_legend: bool = True,
) -> dict[str, int]:
    """Draw TP / FP / FN boxes on *ax* and return counts.

    Parameters
    ----------
    ax : Axes
        Target matplotlib axes.
    predictions : list[OBBParams]
        Predicted boxes.
    ground_truth : list[OBBParams]
        Instance-derived proxy reference boxes.
    matched_pred_indices : set[int]
        Indices into *predictions* that are true positives.
    matched_gt_indices : set[int]
        Indices into *ground_truth* that are true positives.
    show_gt : bool
        When True, draw GT reference outlines (blue dotted) for unmatched GT.
    show_legend : bool
        When True, attach a legend to the axes.

    Returns
    -------
    dict with keys ``tp``, ``fp``, ``fn``.
    """
    tp_count = 0
    fp_count = 0
    fn_count = 0

    # ── TP: matched predictions ──────────────────────────────────────────
    first_tp = True
    for idx in sorted(matched_pred_indices):
        _add_box_patch(
            ax,
            predictions[idx],
            edgecolor=COLOR_TP,
            linestyle="-",
            fill=True,
            fill_alpha=_FILL_ALPHA,
            label="TP" if first_tp else None,
        )
        tp_count += 1
        first_tp = False

    # ── FP: unmatched predictions ────────────────────────────────────────
    first_fp = True
    for idx in range(len(predictions)):
        if idx in matched_pred_indices:
            continue
        _add_box_patch(
            ax,
            predictions[idx],
            edgecolor=COLOR_FP,
            linestyle="-",
            fill=True,
            fill_alpha=_FILL_ALPHA,
            label="FP" if first_fp else None,
        )
        fp_count += 1
        first_fp = False

    # ── FN: unmatched GT ─────────────────────────────────────────────────
    first_fn = True
    for idx in range(len(ground_truth)):
        if idx in matched_gt_indices:
            continue
        _add_box_patch(
            ax,
            ground_truth[idx],
            edgecolor=COLOR_FN,
            linestyle="--",
            linewidth=1.4,
            label="FN" if first_fn else None,
        )
        fn_count += 1
        first_fn = False

    # ── GT reference (blue dotted) for unmatched GT ──────────────────────
    if show_gt:
        first_gt = True
        for idx in range(len(ground_truth)):
            if idx in matched_gt_indices:
                continue
            _add_box_patch(
                ax,
                ground_truth[idx],
                edgecolor=COLOR_GT,
                linestyle=":",
                linewidth=0.8,
                label="GT ref" if first_gt else None,
            )
            first_gt = False

    if show_legend:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    return {"tp": tp_count, "fp": fp_count, "fn": fn_count}


def draw_metrics_text(
    ax: Axes,
    tp: int,
    fp: int,
    fn: int,
    precision: float,
    recall: float,
    f1: float,
    frame_id: int,
    *,
    distance_str: str | None = None,
) -> None:
    """Annotate *ax* with per-frame detection metrics.

    Text is placed in the upper-left corner of the axes.
    """
    lines = [
        f"frame {frame_id}",
        f"TP={tp}  FP={fp}  FN={fn}",
        f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}",
    ]
    if distance_str is not None:
        lines.append(distance_str)

    text = "\n".join(lines)
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
    )


def draw_distance_scale(ax: Axes, roi_x_range: tuple[float, float]) -> None:
    """Draw distance tick marks along the x axis at 10 m intervals."""
    x_min, x_max = roi_x_range
    y_min, y_max = ax.get_ylim()

    first_tick = int(np.ceil(x_min / 10.0)) * 10
    for d in range(first_tick, int(x_max) + 1, 10):
        if d < x_min:
            continue
        ax.axvline(d, color="#cccccc", linewidth=0.4, linestyle=":", zorder=0)
        ax.text(
            d,
            y_min + 0.5,
            f"{d}m",
            ha="center",
            va="bottom",
            fontsize=6,
            color="#888888",
        )
