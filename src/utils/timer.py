"""
Timer utilities for per-module timing and statistics.

Provides:
- Timer context manager for measuring execution time
- @timed decorator for automatic function timing
- TimingStats for aggregating statistics across frames
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Callable
from typing import Optional

import numpy as np


@dataclass
class TimingRecord:
    """Single timing measurement."""

    module: str
    frame_id: int
    duration_ms: float
    input_points: int | None = None
    output_points: int | None = None


@dataclass
class TimingStats:
    """
    Aggregated timing statistics for a module.

    Computes P50, P90, P99 percentiles and other statistics.
    """

    module: str
    durations_ms: list[float] = field(default_factory=list)

    def add(self, duration_ms: float) -> None:
        """Add a timing measurement."""
        self.durations_ms.append(duration_ms)

    @property
    def count(self) -> int:
        return len(self.durations_ms)

    @property
    def mean(self) -> float:
        return float(np.mean(self.durations_ms)) if self.durations_ms else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.durations_ms)) if self.durations_ms else 0.0

    @property
    def p50(self) -> float:
        return float(np.percentile(self.durations_ms, 50)) if self.durations_ms else 0.0

    @property
    def p90(self) -> float:
        return float(np.percentile(self.durations_ms, 90)) if self.durations_ms else 0.0

    @property
    def p99(self) -> float:
        return float(np.percentile(self.durations_ms, 99)) if self.durations_ms else 0.0

    @property
    def min(self) -> float:
        return float(np.min(self.durations_ms)) if self.durations_ms else 0.0

    @property
    def max(self) -> float:
        return float(np.max(self.durations_ms)) if self.durations_ms else 0.0

    @property
    def total(self) -> float:
        return float(np.sum(self.durations_ms)) if self.durations_ms else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "module": self.module,
            "count": self.count,
            "mean_ms": round(self.mean, 3),
            "std_ms": round(self.std, 3),
            "p50_ms": round(self.p50, 3),
            "p90_ms": round(self.p90, 3),
            "p99_ms": round(self.p99, 3),
            "min_ms": round(self.min, 3),
            "max_ms": round(self.max, 3),
            "total_ms": round(self.total, 3),
        }

    def __str__(self) -> str:
        return (
            f"{self.module}: "
            f"mean={self.mean:.1f}ms, "
            f"P50={self.p50:.1f}ms, "
            f"P90={self.p90:.1f}ms, "
            f"P99={self.p99:.1f}ms "
            f"(n={self.count})"
        )


class TimingManager:
    """
    Global timing manager for collecting and aggregating timing data.

    Usage:
        manager = TimingManager()

        with manager.measure("preprocess", frame_id=0):
            # ... processing code ...

        # Get statistics
        stats = manager.get_stats("preprocess")
        print(stats.p50)
    """

    _instance: TimingManager | None = None

    def __new__(cls) -> "TimingManager":
        """Singleton pattern for global timing manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._records: list[TimingRecord] = []
        self._stats: dict[str, TimingStats] = {}
        self._enabled: bool = True
        self._initialized = True

    def enable(self) -> None:
        """Enable timing collection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable timing collection."""
        self._enabled = False

    def reset(self) -> None:
        """Clear all timing data."""
        self._records.clear()
        self._stats.clear()

    def measure(
        self,
        module: str,
        frame_id: int = -1,
        input_points: Optional[int] = None,
    ) -> "Timer":
        """
        Create a Timer context manager for measuring execution time.

        Args:
            module: Name of the module being timed
            frame_id: Frame index (for per-frame tracking)
            input_points: Number of input points (optional)

        Returns:
            Timer context manager
        """
        return Timer(
            module=module,
            frame_id=frame_id,
            input_points=input_points,
            manager=self if self._enabled else None,
        )

    def record(self, record: TimingRecord) -> None:
        """Add a timing record."""
        if not self._enabled:
            return
        self._records.append(record)
        self._stats.setdefault(record.module, TimingStats(record.module)).add(
            record.duration_ms
        )

    def get_stats(self, module: str) -> TimingStats | None:
        """Get timing statistics for a module."""
        return self._stats.get(module)

    def get_all_stats(self) -> dict[str, TimingStats]:
        """Get timing statistics for all modules."""
        return dict(self._stats)

    def get_records(self) -> list[TimingRecord]:
        """Get all timing records."""
        return self._records.copy()

    def summary(self) -> str:
        """Generate a summary string of all timing statistics."""
        lines = ["=" * 60, "Timing Summary", "=" * 60]
        for module, stats in sorted(self._stats.items()):
            lines.append(str(stats))
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export all statistics as dictionary."""
        return {module: stats.to_dict() for module, stats in self._stats.items()}


class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer("my_module", frame_id=0) as t:
            # ... code to time ...
        print(f"Took {t.duration_ms:.2f}ms")
    """

    def __init__(
        self,
        module: str,
        frame_id: int = -1,
        input_points: int | None = None,
        manager: TimingManager | None = None,
    ) -> None:
        self.module = module
        self.frame_id = frame_id
        self.input_points = input_points
        self.output_points: int | None = None
        self._manager = manager
        self._start_time: float = 0.0
        self._end_time: float = 0.0

    def __enter__(self) -> "Timer":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self._end_time = time.perf_counter()
        if self._manager is not None:
            record = TimingRecord(
                module=self.module,
                frame_id=self.frame_id,
                duration_ms=self.duration_ms,
                input_points=self.input_points,
                output_points=self.output_points,
            )
            self._manager.record(record)

    def set_output_points(self, n: int) -> None:
        """Set the number of output points (call before exiting context)."""
        self.output_points = n

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return (self._end_time - self._start_time) * 1000.0


def timed(
    module: str | None = None,
    log: bool = True,
) -> Callable:
    """
    Decorator for timing function execution.

    Args:
        module: Module name (defaults to function name)
        log: Whether to log timing to the global manager

    Usage:
        @timed("my_module")
        def process_frame(points):
            # ... processing ...
            return result
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            module_name: str = module if module is not None else func.__name__
            manager = TimingManager() if log else None
            with Timer(module=module_name, manager=manager):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Global timing manager instance
_timing_manager: TimingManager | None = None


def get_timing_manager() -> TimingManager:
    """Get the global timing manager instance."""
    global _timing_manager
    if _timing_manager is None:
        _timing_manager = TimingManager()
    return _timing_manager
