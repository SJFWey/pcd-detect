"""
Logging utilities for pcd-detect.

Provides:
- Consistent log formatting across modules
- Per-frame logging with timing and point count information
- Log file management
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Custom log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Frame-specific format for pipeline logging
FRAME_FORMAT = "[Frame {frame:06d}] {module}: {message}"
POINTS_FORMAT = "[Frame {frame:06d}] {module}: {input_pts:,} -> {output_pts:,} pts ({reduction:.1f}% reduction) in {time_ms:.1f}ms"


class FrameLogAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds frame context to log messages.

    Usage:
        logger = get_logger("preprocess")
        frame_logger = FrameLogAdapter(logger, frame_id=0)
        frame_logger.info("Processing started")
    """

    def __init__(self, logger: logging.Logger, frame_id: int = -1):
        super().__init__(logger, {"frame_id": frame_id})
        self.frame_id = frame_id

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if self.frame_id >= 0:
            msg = f"[Frame {self.frame_id:06d}] {msg}"
        return msg, kwargs

    def set_frame(self, frame_id: int) -> None:
        """Update the current frame ID."""
        self.frame_id = frame_id
        self.extra["frame_id"] = frame_id


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for terminal output.

    Colors:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red + Bold
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def __init__(
        self,
        fmt: str = LOG_FORMAT,
        datefmt: str = LOG_DATE_FORMAT,
        use_colors: bool = True,
    ) -> None:
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    use_colors: bool = True,
) -> None:
    """
    Set up logging configuration for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        use_colors: Whether to use colored output in terminal
    """
    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger("pcd_detect")
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(ColoredFormatter(use_colors=use_colors))
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        root_logger.addHandler(file_handler)

    # Suppress verbose third-party loggers
    logging.getLogger("open3d").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (e.g., "preprocess.ground")

    Returns:
        Logger instance
    """
    return logging.getLogger(f"pcd_detect.{name}")


def log_frame_processing(
    logger: logging.Logger,
    frame_id: int,
    module: str,
    input_points: int,
    output_points: int,
    duration_ms: float,
) -> None:
    """
    Log frame processing statistics.

    Args:
        logger: Logger instance
        frame_id: Frame index
        module: Module name
        input_points: Number of input points
        output_points: Number of output points
        duration_ms: Processing time in milliseconds
    """
    if input_points > 0:
        reduction = (1.0 - output_points / input_points) * 100.0
    else:
        reduction = 0.0

    logger.info(
        POINTS_FORMAT.format(
            frame=frame_id,
            module=module,
            input_pts=input_points,
            output_pts=output_points,
            reduction=reduction,
            time_ms=duration_ms,
        )
    )


def create_run_log_path(output_dir: Path, prefix: str = "run") -> Path:
    """
    Create a timestamped log file path.

    Args:
        output_dir: Output directory for logs
        prefix: Log file prefix

    Returns:
        Path to log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{prefix}_{timestamp}.log"
