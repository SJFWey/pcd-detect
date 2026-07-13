"""
Common utilities shared across command modules.
"""

from pathlib import Path
from typing import Any, Mapping

import typer

from ..utils.config import load_config as load_config_file


def load_config(config_path: Path | None, task: str) -> dict[str, Any]:
    """Load configuration for a specific task.

    Args:
        config_path: Explicit config file path (legacy mode) or None for modular loading.
        task: Task type for automatic config merging (run, eval, viz, export).
    """
    try:
        if config_path is not None:
            return load_config_file(config_path)
        return load_config_file(task=task)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def print_config(config: Mapping[str, Any], indent: int = 0) -> None:
    """Pretty print configuration dictionary."""
    for key, value in config.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            typer.echo(f"{prefix}{key}:")
            print_config(value, indent + 1)
        else:
            typer.echo(f"{prefix}{key}: {value}")


def parse_frame_range(frames_str: str | None, num_frames: int) -> range:
    """Parse frame range string like '0-100' or '50' into a range."""
    if frames_str is None:
        return range(num_frames)

    value = frames_str.strip()
    if not value:
        raise typer.BadParameter("frames must not be empty")

    try:
        if "-" in value:
            parts = value.split("-")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError
            start = int(parts[0])
            end = min(int(parts[1]), num_frames)
            return range(start, end)

        # Single frame
        frame_idx = int(value)
    except ValueError as exc:
        raise typer.BadParameter(
            "frames must be a single frame or range like '50' or '0-100'"
        ) from exc

    if frame_idx < 0:
        raise typer.BadParameter("frames must start at >= 0")
    return range(frame_idx, min(frame_idx + 1, num_frames))


def normalize_target_name(name: str) -> str:
    """Normalize target class names from config."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "pedestrian": "person",
        "cyclist": "bicyclist",
    }
    return aliases.get(key, key)
