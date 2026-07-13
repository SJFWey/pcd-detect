"""
Configuration loader for pcd-detect.

Responsibilities:
- Load modular YAML configs from configs/ directory.
- Merge configs based on task (run, eval, viz, export).
- Apply optional local overrides via config.local.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

# Config file locations
CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"
LOCAL_CONFIG_NAME = "config.local.yaml"

# Task-specific config mappings
TASK_CONFIGS = {
    "run": ["base.yaml", "detection.yaml"],
    "eval": ["base.yaml", "detection.yaml", "evaluation.yaml"],
    "viz": ["base.yaml", "detection.yaml", "visualization.yaml"],
    "export": ["base.yaml", "detection.yaml", "export.yaml", "visualization.yaml"],
    "tools": [
        "base.yaml",
        "detection.yaml",
        "evaluation.yaml",
        "export.yaml",
        "visualization.yaml",
    ],
}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")
    return data


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge override into base, returning base for convenience."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    config_path: Path | str | None = None,
    task: str | None = None,
    local_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load YAML config with task-based merging and local overrides.

    Args:
        config_path: Explicit config file path. If provided, loads only this file
                     (legacy mode for backwards compatibility).
        task: Task type for automatic config merging (run, eval, viz, export, tools).
              If None and config_path is None, defaults to loading all configs.
        local_path: Path to local override file. If None, looks for config.local.yaml.

    Returns:
        Merged configuration dictionary.
    """
    config: dict[str, Any] = {}

    # Legacy mode: load single explicit config file
    if config_path is not None:
        base_path = Path(config_path)
        config = _load_yaml(base_path)
    else:
        # Modular mode: load configs based on task
        config_files = TASK_CONFIGS.get(task or "tools", list(TASK_CONFIGS["tools"]))

        for filename in config_files:
            file_path = CONFIGS_DIR / filename
            if file_path.exists():
                file_config = _load_yaml(file_path)
                config = deep_merge(config, file_config)

    # Apply local overrides
    if local_path is None:
        local_path = CONFIGS_DIR / LOCAL_CONFIG_NAME
    else:
        local_path = Path(local_path)

    if local_path.exists():
        local_cfg = _load_yaml(local_path)
        config = deep_merge(config, local_cfg)

    return config


def load_config_for_task(task: str) -> dict[str, Any]:
    """Convenience function to load config for a specific task.

    Args:
        task: One of 'run', 'eval', 'viz', 'export', 'tools'.

    Returns:
        Merged configuration dictionary for the task.
    """
    return load_config(task=task)
