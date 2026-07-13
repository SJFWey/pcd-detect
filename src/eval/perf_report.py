"""Performance reporting module.

This module generates performance reports with timing statistics:
- Per-module timing breakdown
- P50, P90, P99 percentiles
- Total processing time per frame
- Memory usage estimates

Usage:
    from src.utils.timer import get_timing_manager
    from src.eval.perf_report import generate_performance_report

    # After running the pipeline...
    manager = get_timing_manager()
    report = generate_performance_report(manager)
    report.save("outputs/reports/perf.json")
    report.save_markdown("outputs/reports/perf.md")
"""

import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.logging import get_logger
from ..utils.timer import TimingManager, TimingStats

logger = get_logger(__name__)


@dataclass
class ModulePerformance:
    """Performance statistics for a single module."""

    name: str
    count: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    total_ms: float
    percentage: float = 0.0  # Percentage of total time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "percentage": round(self.percentage, 2),
        }

    @classmethod
    def from_timing_stats(
        cls, stats: TimingStats, total_time: float = 0.0
    ) -> "ModulePerformance":
        """Create from TimingStats."""
        percentage = (stats.total / total_time * 100) if total_time > 0 else 0.0
        return cls(
            name=stats.module,
            count=stats.count,
            mean_ms=stats.mean,
            std_ms=stats.std,
            p50_ms=stats.p50,
            p90_ms=stats.p90,
            p99_ms=stats.p99,
            min_ms=stats.min,
            max_ms=stats.max,
            total_ms=stats.total,
            percentage=percentage,
        )


@dataclass
class FrameTimingStats:
    """Per-frame timing statistics."""

    mean_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    fps_mean: float
    fps_p50: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "fps_mean": round(self.fps_mean, 2),
            "fps_p50": round(self.fps_p50, 2),
        }


@dataclass
class PerformanceReport:
    """Complete performance report.

    Attributes:
        timestamp: Report generation timestamp.
        system_info: System information (OS, Python version, etc.).
        modules: Per-module performance statistics.
        frame_timing: Per-frame aggregate timing.
        total_frames: Total number of frames processed.
        total_time_ms: Total processing time in milliseconds.
        total_points: Total number of points processed.
    """

    timestamp: str
    system_info: dict[str, str]
    modules: list[ModulePerformance]
    frame_timing: FrameTimingStats | None
    total_frames: int
    total_time_ms: float
    total_points: int = 0
    points_per_second: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "modules": [m.to_dict() for m in self.modules],
            "frame_timing": self.frame_timing.to_dict() if self.frame_timing else None,
            "summary": {
                "total_frames": self.total_frames,
                "total_time_ms": round(self.total_time_ms, 3),
                "total_points": self.total_points,
                "points_per_second": round(self.points_per_second, 0),
            },
        }

    def save(self, output_path: str | Path) -> None:
        """Save report to JSON file.

        Args:
            output_path: Path to output JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved performance report to {output_path}")

    def save_markdown(self, output_path: str | Path) -> None:
        """Save report as Markdown file.

        Args:
            output_path: Path to output Markdown file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = self._generate_markdown()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Saved performance report (Markdown) to {output_path}")

    def _generate_markdown(self) -> list[str]:
        """Generate Markdown report content."""
        lines = [
            "# Performance Report",
            "",
            f"Generated: {self.timestamp}",
            "",
            "## System Information",
            "",
        ]

        for key, value in self.system_info.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

        # Summary
        lines.extend(
            [
                "## Summary",
                "",
                f"- **Total Frames**: {self.total_frames}",
                f"- **Total Time**: {self.total_time_ms / 1000:.2f}s ({self.total_time_ms:.1f}ms)",
                f"- **Total Points**: {self.total_points:,}",
                f"- **Points/Second**: {self.points_per_second:,.0f}",
                "",
            ]
        )

        # Frame timing
        if self.frame_timing:
            ft = self.frame_timing
            lines.extend(
                [
                    "## Per-Frame Timing",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| Mean | {ft.mean_ms:.2f} ms |",
                    f"| Std | {ft.std_ms:.2f} ms |",
                    f"| P50 (median) | {ft.p50_ms:.2f} ms |",
                    f"| P90 | {ft.p90_ms:.2f} ms |",
                    f"| P99 | {ft.p99_ms:.2f} ms |",
                    f"| Min | {ft.min_ms:.2f} ms |",
                    f"| Max | {ft.max_ms:.2f} ms |",
                    f"| FPS (mean) | {ft.fps_mean:.1f} |",
                    f"| FPS (P50) | {ft.fps_p50:.1f} |",
                    "",
                ]
            )

        # Module breakdown
        if self.modules:
            lines.extend(
                [
                    "## Module Breakdown",
                    "",
                    "| Module | Mean (ms) | P50 (ms) | P90 (ms) | P99 (ms) | % Time |",
                    "|--------|-----------|----------|----------|----------|--------|",
                ]
            )

            # Sort by total time descending
            sorted_modules = sorted(
                self.modules, key=lambda m: m.total_ms, reverse=True
            )
            for m in sorted_modules:
                lines.append(
                    f"| {m.name} | {m.mean_ms:.2f} | {m.p50_ms:.2f} | "
                    f"{m.p90_ms:.2f} | {m.p99_ms:.2f} | {m.percentage:.1f}% |"
                )
            lines.append("")

            # Detailed statistics
            lines.extend(
                [
                    "### Detailed Module Statistics",
                    "",
                ]
            )

            for m in sorted_modules:
                lines.extend(
                    [
                        f"#### {m.name}",
                        "",
                        f"- Count: {m.count}",
                        f"- Mean: {m.mean_ms:.2f} ms (±{m.std_ms:.2f})",
                        f"- Range: [{m.min_ms:.2f}, {m.max_ms:.2f}] ms",
                        f"- Percentiles: P50={m.p50_ms:.2f}, P90={m.p90_ms:.2f}, P99={m.p99_ms:.2f} ms",
                        f"- Total: {m.total_ms:.2f} ms ({m.percentage:.1f}% of total)",
                        "",
                    ]
                )

        return lines

    def print_summary(self) -> None:
        """Print a summary to console."""
        print("\n" + "=" * 60)
        print("Performance Report Summary")
        print("=" * 60)

        print(f"\nTotal Frames: {self.total_frames}")
        print(f"Total Time: {self.total_time_ms / 1000:.2f}s")
        print(f"Total Points: {self.total_points:,}")

        if self.frame_timing:
            ft = self.frame_timing
            print("\nPer-Frame Timing:")
            print(f"  Mean: {ft.mean_ms:.2f} ms")
            print(f"  P50:  {ft.p50_ms:.2f} ms")
            print(f"  P90:  {ft.p90_ms:.2f} ms")
            print(f"  P99:  {ft.p99_ms:.2f} ms")
            print(f"  FPS:  {ft.fps_mean:.1f} (mean), {ft.fps_p50:.1f} (P50)")

        if self.modules:
            print("\nModule Breakdown:")
            sorted_modules = sorted(
                self.modules, key=lambda m: m.total_ms, reverse=True
            )
            for m in sorted_modules[:10]:  # Top 10 modules
                print(
                    f"  {m.name:20s}: {m.mean_ms:8.2f} ms (P99: {m.p99_ms:8.2f} ms) [{m.percentage:5.1f}%]"
                )

        print("=" * 60)


def get_system_info() -> dict[str, str]:
    """Get system information for the report."""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "Unknown",
        "architecture": platform.machine(),
    }


def compute_frame_timing(
    manager: TimingManager,
    frame_module: str = "total",
) -> FrameTimingStats | None:
    """Compute per-frame timing statistics.

    Args:
        manager: TimingManager with recorded timings.
        frame_module: Module name that represents total frame time.

    Returns:
        FrameTimingStats or None if no data.
    """
    stats = manager.get_stats(frame_module)
    if stats is None or stats.count == 0:
        # Try to compute from all records
        records = manager.get_records()
        if not records:
            return None

        # Group by frame_id and sum durations
        frame_times: dict[int, float] = {}
        for record in records:
            if record.frame_id >= 0:
                frame_times[record.frame_id] = (
                    frame_times.get(record.frame_id, 0.0) + record.duration_ms
                )

        if not frame_times:
            return None

        durations = list(frame_times.values())
    else:
        durations = stats.durations_ms

    if not durations:
        return None

    arr = np.array(durations)
    mean_ms = float(np.mean(arr))
    p50_ms = float(np.percentile(arr, 50))

    return FrameTimingStats(
        mean_ms=mean_ms,
        std_ms=float(np.std(arr)),
        p50_ms=p50_ms,
        p90_ms=float(np.percentile(arr, 90)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        fps_mean=1000.0 / mean_ms if mean_ms > 0 else 0.0,
        fps_p50=1000.0 / p50_ms if p50_ms > 0 else 0.0,
    )


def generate_performance_report(
    manager: TimingManager,
    total_points: int = 0,
    frame_module: str = "total",
) -> PerformanceReport:
    """Generate a complete performance report from timing data.

    Args:
        manager: TimingManager with recorded timings.
        total_points: Total number of points processed.
        frame_module: Module name for per-frame timing.

    Returns:
        PerformanceReport with all statistics.
    """
    all_stats = manager.get_all_stats()

    # Compute total time across all modules
    total_time_ms = sum(s.total for s in all_stats.values())

    # Create module performance objects
    modules = [
        ModulePerformance.from_timing_stats(stats, total_time_ms)
        for stats in all_stats.values()
    ]

    # Compute frame timing
    frame_timing = compute_frame_timing(manager, frame_module)

    # Get frame count
    total_frames = 0
    if frame_timing:
        stats = manager.get_stats(frame_module)
        if stats:
            total_frames = stats.count
    if total_frames == 0:
        # Estimate from records
        records = manager.get_records()
        frame_ids = {r.frame_id for r in records if r.frame_id >= 0}
        total_frames = len(frame_ids) if frame_ids else 0

    # Compute points per second
    points_per_second = 0.0
    if total_time_ms > 0 and total_points > 0:
        points_per_second = total_points / (total_time_ms / 1000.0)

    return PerformanceReport(
        timestamp=datetime.now().isoformat(),
        system_info=get_system_info(),
        modules=modules,
        frame_timing=frame_timing,
        total_frames=total_frames,
        total_time_ms=total_time_ms,
        total_points=total_points,
        points_per_second=points_per_second,
    )


def load_performance_report(path: str | Path) -> PerformanceReport:
    """Load a performance report from JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        PerformanceReport object.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modules = [
        ModulePerformance(
            name=m["name"],
            count=m["count"],
            mean_ms=m["mean_ms"],
            std_ms=m["std_ms"],
            p50_ms=m["p50_ms"],
            p90_ms=m["p90_ms"],
            p99_ms=m["p99_ms"],
            min_ms=m["min_ms"],
            max_ms=m["max_ms"],
            total_ms=m["total_ms"],
            percentage=m.get("percentage", 0.0),
        )
        for m in data.get("modules", [])
    ]

    frame_timing = None
    if data.get("frame_timing"):
        ft = data["frame_timing"]
        frame_timing = FrameTimingStats(
            mean_ms=ft["mean_ms"],
            std_ms=ft["std_ms"],
            p50_ms=ft["p50_ms"],
            p90_ms=ft["p90_ms"],
            p99_ms=ft["p99_ms"],
            min_ms=ft["min_ms"],
            max_ms=ft["max_ms"],
            fps_mean=ft["fps_mean"],
            fps_p50=ft["fps_p50"],
        )

    summary = data.get("summary", {})

    return PerformanceReport(
        timestamp=data.get("timestamp", ""),
        system_info=data.get("system_info", {}),
        modules=modules,
        frame_timing=frame_timing,
        total_frames=summary.get("total_frames", 0),
        total_time_ms=summary.get("total_time_ms", 0.0),
        total_points=summary.get("total_points", 0),
        points_per_second=summary.get("points_per_second", 0.0),
    )
