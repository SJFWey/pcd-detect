"""src.datasets.label_maps

Utilities for SemanticKITTI label remapping.

This module reuses the official SemanticKITTI configuration
(`third_party/semkitti_api/config/semantic-kitti.yaml`) to guarantee that
`learning_map` / `learning_map_inv` match the official evaluation scripts.
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray


def _default_semkitti_config_path() -> Path:
    third_party = Path(__file__).resolve().parents[2] / "third_party" / "semkitti_api"
    return third_party / "config" / "semantic-kitti.yaml"


def _resolve_semkitti_config_path(config_path: str | Path | None) -> Path:
    if config_path is None:
        return _default_semkitti_config_path()
    return Path(config_path)


def load_semkitti_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the official SemanticKITTI YAML config."""
    path = _resolve_semkitti_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"SemanticKITTI config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_lut(
    mapping: dict[int, int], *, default: int = 0, dtype=np.int32
) -> NDArray[np.int32]:
    """Build a dense lookup table from an int->int mapping."""
    if not mapping:
        return np.zeros((0,), dtype=dtype)
    max_key = int(max(mapping.keys()))
    lut = np.full((max_key + 1,), int(default), dtype=dtype)
    for k, v in mapping.items():
        lut[int(k)] = int(v)
    return lut


def remap_with_lut(
    values: NDArray[np.integer], lut: NDArray[np.integer]
) -> NDArray[np.int32]:
    """Remap integer labels using a LUT.

    Raises a clear error if values exceed LUT bounds.
    """
    if lut.size == 0:
        raise ValueError("LUT is empty; cannot remap values")
    if values.size == 0:
        return values.astype(np.int32, copy=False)

    max_val = int(np.max(values))
    if max_val >= lut.shape[0]:
        raise ValueError(
            f"Label value {max_val} out of LUT range [0, {lut.shape[0] - 1}]"
        )

    idx = values.astype(np.intp, copy=False)
    return lut[idx].astype(np.int32, copy=False)


@dataclass
class SemanticKITTIMaps:
    """Holds SemanticKITTI remapping tables (learning_map and inverse)."""

    config_path: Path
    labels: dict[int, str]
    learning_map: dict[int, int]
    learning_map_inv: dict[int, int]
    learning_ignore: dict[int, bool]

    learning_map_lut: NDArray[np.int32]
    learning_map_inv_lut: NDArray[np.int32]

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "SemanticKITTIMaps":
        path = _resolve_semkitti_config_path(config_path)
        cfg = load_semkitti_config(path)

        # YAML keys are expected to be ints; cast defensively.
        labels = {int(k): str(v) for k, v in cfg.get("labels", {}).items()}
        learning_map = {int(k): int(v) for k, v in cfg.get("learning_map", {}).items()}
        learning_map_inv = {
            int(k): int(v) for k, v in cfg.get("learning_map_inv", {}).items()
        }
        learning_ignore = {
            int(k): bool(v) for k, v in cfg.get("learning_ignore", {}).items()
        }

        return cls(
            config_path=path,
            labels=labels,
            learning_map=learning_map,
            learning_map_inv=learning_map_inv,
            learning_ignore=learning_ignore,
            learning_map_lut=build_lut(learning_map, default=0, dtype=np.int32),
            learning_map_inv_lut=build_lut(learning_map_inv, default=0, dtype=np.int32),
        )

    def map_to_training(self, semantic_ids: NDArray[np.integer]) -> NDArray[np.int32]:
        return remap_with_lut(semantic_ids, self.learning_map_lut)

    def map_from_training(self, training_ids: NDArray[np.integer]) -> NDArray[np.int32]:
        return remap_with_lut(training_ids, self.learning_map_inv_lut)
