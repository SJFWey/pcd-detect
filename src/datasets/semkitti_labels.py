"""
SemanticKITTI label utilities.

This module provides utilities for working with SemanticKITTI labels,
including semantic class mappings, instance ID extraction, and color maps.
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Versioned project-owned SemanticKITTI mapping used by detection evaluation.
# The optional official SemanticKITTI API remains responsible for official metrics.
SEMKITTI_CONFIG_PATH = (
    Path(__file__).resolve().parent / "resources" / "semkitti_detection_labels.yaml"
)


class SemanticKITTILabels:
    """
    Utility class for working with SemanticKITTI labels.

    Provides:
    - Semantic class ID to name mapping
    - Instance ID extraction from combined labels
    - Learning map for training (remaps classes)
    - Inverse learning map for prediction output
    - Color map for visualization

    Label Format:
        Each label is a 32-bit unsigned integer:
        - Lower 16 bits: semantic class ID
        - Upper 16 bits: instance ID

    Attributes:
        labels: Dict mapping semantic ID to class name.
        color_map: Dict mapping semantic ID to BGR color tuple.
        learning_map: Dict mapping original ID to training ID.
        learning_map_inv: Dict mapping training ID to original ID.
        learning_ignore: Dict marking which training IDs to ignore.
        split: Dict containing train/valid/test sequence lists.
    """

    # Classes that represent "things" (objects with instances)
    THING_CLASSES = {
        10: "car",
        11: "bicycle",
        13: "bus",
        15: "motorcycle",
        16: "on-rails",
        18: "truck",
        20: "other-vehicle",
        30: "person",
        31: "bicyclist",
        32: "motorcyclist",
        252: "moving-car",
        253: "moving-bicyclist",
        254: "moving-person",
        255: "moving-motorcyclist",
        256: "moving-on-rails",
        257: "moving-bus",
        258: "moving-truck",
        259: "moving-other-vehicle",
    }

    # Classes that represent "stuff" (no instances)
    STUFF_CLASSES = {
        0: "unlabeled",
        1: "outlier",
        40: "road",
        44: "parking",
        48: "sidewalk",
        49: "other-ground",
        50: "building",
        51: "fence",
        52: "other-structure",
        60: "lane-marking",
        70: "vegetation",
        71: "trunk",
        72: "terrain",
        80: "pole",
        81: "traffic-sign",
        99: "other-object",
    }

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize SemanticKITTI label utilities.

        Args:
            config_path: Path to the project-owned detection label mapping.
                         The optional official SemanticKITTI API configuration is
                         used only by the official evaluator.
        """

        if config_path is None:
            config_path = SEMKITTI_CONFIG_PATH

        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self._config = self._load_config()
        self._setup_mappings()

        logger.info(f"Loaded SemanticKITTI config from {self.config_path}")

    def _load_config(self) -> dict[str, Any]:
        """Load the semantic-kitti.yaml config file."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _setup_mappings(self) -> None:
        """Setup label mappings from config."""
        self.labels: dict[int, str] = self._config.get("labels", {})
        self.color_map: dict[int, tuple[int, int, int]] = self._config.get(
            "color_map", {}
        )
        self.learning_map: dict[int, int] = self._config.get("learning_map", {})
        self.learning_map_inv: dict[int, int] = self._config.get("learning_map_inv", {})
        self.learning_ignore: dict[int, bool] = self._config.get("learning_ignore", {})
        self.split: dict[str, list[int]] = self._config.get("split", {})

        # Build numpy lookup tables for fast mapping
        max_label = max(self.learning_map.keys()) + 1
        self._learning_map_lut = np.zeros((max_label,), dtype=np.int32)
        for k, v in self.learning_map.items():
            self._learning_map_lut[k] = v

        max_training_id = max(self.learning_map_inv.keys()) + 1
        self._learning_map_inv_lut = np.zeros((max_training_id,), dtype=np.int32)
        for k, v in self.learning_map_inv.items():
            self._learning_map_inv_lut[k] = v

    @staticmethod
    def extract_semantic(labels: NDArray[np.uint32]) -> NDArray[np.uint16]:
        """
        Extract semantic class IDs from combined labels.

        Args:
            labels: (N,) array of uint32 combined labels.

        Returns:
            (N,) array of uint16 semantic class IDs.
        """
        return (labels & 0xFFFF).astype(np.uint16)

    @staticmethod
    def extract_instance(labels: NDArray[np.uint32]) -> NDArray[np.uint16]:
        """
        Extract instance IDs from combined labels.

        Args:
            labels: (N,) array of uint32 combined labels.

        Returns:
            (N,) array of uint16 instance IDs.
        """
        return (labels >> 16).astype(np.uint16)

    @staticmethod
    def combine_labels(
        semantic: NDArray[np.uint16], instance: NDArray[np.uint16]
    ) -> NDArray[np.uint32]:
        """
        Combine semantic and instance IDs into a single label.

        Args:
            semantic: (N,) array of semantic class IDs.
            instance: (N,) array of instance IDs.

        Returns:
            (N,) array of combined uint32 labels.
        """
        return semantic.astype(np.uint32) | (instance.astype(np.uint32) << 16)

    def map_to_training(self, labels: NDArray[np.uint16]) -> NDArray[np.int32]:
        """
        Map original semantic labels to training labels using learning_map.

        Args:
            labels: (N,) array of original semantic class IDs.

        Returns:
            (N,) array of training class IDs.
        """
        return self._learning_map_lut[labels]

    def map_from_training(self, labels: NDArray[np.int32]) -> NDArray[np.int32]:
        """
        Map training labels back to original labels using learning_map_inv.

        Args:
            labels: (N,) array of training class IDs.

        Returns:
            (N,) array of original semantic class IDs.
        """
        return self._learning_map_inv_lut[labels]

    def get_label_name(self, label_id: int) -> str:
        """Get the name of a semantic class by ID."""
        return self.labels.get(label_id, f"unknown-{label_id}")

    def get_label_color(self, label_id: int) -> tuple[int, int, int]:
        """Get the BGR color for a semantic class by ID."""
        return self.color_map.get(label_id, (0, 0, 0))

    def is_thing(self, label_id: int) -> bool:
        """Check if a semantic class represents a 'thing' (object with instances)."""
        return label_id in self.THING_CLASSES

    def is_stuff(self, label_id: int) -> bool:
        """Check if a semantic class represents 'stuff' (no instances)."""
        return label_id in self.STUFF_CLASSES

    def colorize_labels(
        self, labels: NDArray[np.uint16], bgr: bool = True
    ) -> NDArray[np.uint8]:
        """
        Colorize semantic labels for visualization.

        Args:
            labels: (N,) array of semantic class IDs.
            bgr: If True, return BGR colors; otherwise RGB.

        Returns:
            (N, 3) array of uint8 colors.
        """
        # Build color lookup table
        max_label = max(self.color_map.keys()) + 1
        color_lut = np.zeros((max_label, 3), dtype=np.uint8)
        for label_id, color in self.color_map.items():
            color_lut[label_id] = color if bgr else color[::-1]

        # Map labels to colors
        return color_lut[labels]

    def get_semantic_histogram(self, labels: NDArray[np.uint16]) -> dict[str, int]:
        """
        Compute histogram of semantic classes.

        Args:
            labels: (N,) array of semantic class IDs.

        Returns:
            Dict mapping class name to count.
        """
        unique, counts = np.unique(labels, return_counts=True)
        return {self.get_label_name(int(u)): int(c) for u, c in zip(unique, counts)}

    def get_instance_count(self, inst_labels: NDArray[np.uint16]) -> int:
        """
        Count unique instances (excluding 0 which means no instance).

        Args:
            inst_labels: (N,) array of instance IDs.

        Returns:
            Number of unique instances.
        """
        unique = np.unique(inst_labels)
        return len(unique[unique > 0])

    def get_split_sequences(self, split_name: str) -> list[int]:
        """
        Get sequence numbers for a split (train/valid/test).

        Args:
            split_name: Name of split ("train", "valid", or "test").

        Returns:
            List of sequence numbers.
        """
        return self.split.get(split_name, [])


def read_label_file(filepath: str | Path) -> NDArray[np.uint32]:
    """
    Read a .label file and return raw labels.

    Args:
        filepath: Path to .label file.

    Returns:
        (N,) array of uint32 labels.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Label file not found: {filepath}")
    return np.fromfile(str(filepath), dtype=np.uint32)


def write_label_file(filepath: str | Path, labels: NDArray[np.uint32]) -> None:
    """
    Write labels to a .label file.

    Args:
        filepath: Path to output .label file.
        labels: (N,) array of uint32 labels.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    labels.astype(np.uint32).tofile(str(filepath))
