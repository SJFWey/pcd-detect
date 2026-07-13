"""Color utilities for point cloud visualization.

Provides color schemes for:
- SemanticKITTI semantic classes
- Instance coloring
- Cluster visualization
- Ground/non-ground distinction
"""

import numpy as np
from numpy.typing import NDArray


# SemanticKITTI color map (RGB, 0-255)
# Reference: http://semantic-kitti.org/dataset.html
SEMANTIC_KITTI_COLORS = {
    0: (0, 0, 0),  # unlabeled
    1: (0, 0, 255),  # outlier
    10: (245, 150, 100),  # car
    11: (245, 230, 100),  # bicycle
    13: (250, 80, 100),  # bus
    15: (150, 60, 30),  # motorcycle
    16: (255, 0, 0),  # on-rails
    18: (180, 30, 80),  # truck
    20: (255, 0, 0),  # other-vehicle
    30: (30, 30, 255),  # person
    31: (200, 40, 255),  # bicyclist
    32: (90, 30, 150),  # motorcyclist
    40: (255, 0, 255),  # road
    44: (255, 150, 255),  # parking
    48: (75, 0, 75),  # sidewalk
    49: (75, 0, 175),  # other-ground
    50: (0, 200, 255),  # building
    51: (50, 120, 255),  # fence
    52: (0, 150, 255),  # other-structure
    60: (170, 255, 150),  # lane-marking
    70: (0, 175, 0),  # vegetation
    71: (0, 60, 135),  # trunk
    72: (80, 240, 150),  # terrain
    80: (150, 240, 255),  # pole
    81: (0, 0, 255),  # traffic-sign
    99: (255, 255, 50),  # other-object
    252: (245, 150, 100),  # moving-car
    253: (200, 40, 255),  # moving-bicyclist
    254: (30, 30, 255),  # moving-person
    255: (90, 30, 150),  # moving-motorcyclist
    256: (255, 0, 0),  # moving-on-rails
    257: (250, 80, 100),  # moving-bus
    258: (180, 30, 80),  # moving-truck
    259: (255, 0, 0),  # moving-other-vehicle
}

# Thing classes (instances) in SemanticKITTI
THING_CLASSES = {
    10,
    11,
    13,
    15,
    18,
    20,
    30,
    31,
    32,
    252,
    253,
    254,
    255,
    256,
    257,
    258,
    259,
}

# Default colors for visualization modes
COLOR_GROUND = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # Gray
COLOR_NON_GROUND = np.array([0.2, 0.7, 0.2], dtype=np.float32)  # Green
COLOR_UNLABELED = np.array([0.3, 0.3, 0.3], dtype=np.float32)  # Dark gray
COLOR_BOX_DEFAULT = np.array([1.0, 0.2, 0.2], dtype=np.float32)  # Red
COLOR_GT_BOX = np.array([0.2, 1.0, 0.2], dtype=np.float32)  # Green


def get_semantic_color(semantic_id: int) -> tuple[int, int, int]:
    """Get RGB color for a semantic class ID."""
    return SEMANTIC_KITTI_COLORS.get(semantic_id, (128, 128, 128))


def semantic_colors_to_float(
    semantic_ids: NDArray[np.uint16],
) -> NDArray[np.float32]:
    """Convert semantic IDs to float RGB colors (0-1 range).

    Args:
        semantic_ids: (N,) array of semantic class IDs.

    Returns:
        (N, 3) array of RGB colors in [0, 1] range.
    """
    colors = np.zeros((semantic_ids.shape[0], 3), dtype=np.float32)

    for sem_id, rgb in SEMANTIC_KITTI_COLORS.items():
        mask = semantic_ids == sem_id
        if np.any(mask):
            colors[mask] = np.array(rgb, dtype=np.float32) / 255.0

    # Default color for unknown IDs
    unknown_mask = ~np.isin(semantic_ids, list(SEMANTIC_KITTI_COLORS.keys()))
    colors[unknown_mask] = COLOR_UNLABELED

    return colors


def generate_cluster_colors(num_clusters: int, seed: int = 42) -> NDArray[np.float32]:
    """Generate distinct colors for clusters.

    Args:
        num_clusters: Number of clusters to generate colors for.
        seed: Random seed for reproducibility.

    Returns:
        (num_clusters, 3) array of RGB colors in [0, 1] range.
    """
    if num_clusters == 0:
        return np.zeros((0, 3), dtype=np.float32)

    rng = np.random.default_rng(seed)

    # Use HSV color space for better distinction
    hues = np.linspace(0, 1, num_clusters, endpoint=False)
    rng.shuffle(hues)  # Shuffle to avoid similar adjacent colors

    colors = np.zeros((num_clusters, 3), dtype=np.float32)
    for i, h in enumerate(hues):
        # Convert HSV to RGB
        # S = 0.8, V = 0.9 for vibrant colors
        s, v = 0.8, 0.9
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c

        if h < 1 / 6:
            r, g, b = c, x, 0
        elif h < 2 / 6:
            r, g, b = x, c, 0
        elif h < 3 / 6:
            r, g, b = 0, c, x
        elif h < 4 / 6:
            r, g, b = 0, x, c
        elif h < 5 / 6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        colors[i] = [r + m, g + m, b + m]

    return colors


def instance_colors_to_float(
    instance_ids: NDArray[np.uint16],
    seed: int = 42,
) -> NDArray[np.float32]:
    """Convert instance IDs to distinct colors.

    Args:
        instance_ids: (N,) array of instance IDs.
        seed: Random seed for color generation.

    Returns:
        (N, 3) array of RGB colors in [0, 1] range.
    """
    unique_ids = np.unique(instance_ids[instance_ids > 0])
    num_instances = len(unique_ids)

    if num_instances == 0:
        return np.full((instance_ids.shape[0], 3), 0.5, dtype=np.float32)

    cluster_colors = generate_cluster_colors(num_instances, seed)

    colors = np.full((instance_ids.shape[0], 3), 0.5, dtype=np.float32)
    for i, inst_id in enumerate(unique_ids):
        mask = instance_ids == inst_id
        colors[mask] = cluster_colors[i]

    return colors


def colorize_by_height(
    points: NDArray[np.float32],
    z_min: float = -2.0,
    z_max: float = 2.0,
) -> NDArray[np.float32]:
    """Color points by height using a rainbow colormap.

    Args:
        points: (N, 3+) point cloud.
        z_min: Minimum height for color mapping.
        z_max: Maximum height for color mapping.

    Returns:
        (N, 3) RGB colors in [0, 1] range.
    """
    z = points[:, 2]
    z_norm = np.clip((z - z_min) / (z_max - z_min), 0, 1)

    # Rainbow colormap: blue -> cyan -> green -> yellow -> red
    colors = np.zeros((len(z), 3), dtype=np.float32)

    # Red channel
    colors[:, 0] = np.where(z_norm < 0.5, 0, 2 * (z_norm - 0.5))

    # Green channel
    colors[:, 1] = np.where(
        z_norm < 0.25,
        4 * z_norm,
        np.where(z_norm < 0.75, 1.0, 4 * (1 - z_norm)),
    )

    # Blue channel
    colors[:, 2] = np.where(z_norm < 0.5, 1 - 2 * z_norm, 0)

    return colors


def colorize_by_distance(
    points: NDArray[np.float32],
    d_min: float = 0.0,
    d_max: float = 50.0,
) -> NDArray[np.float32]:
    """Color points by distance from origin.

    Args:
        points: (N, 3+) point cloud.
        d_min: Minimum distance for color mapping.
        d_max: Maximum distance for color mapping.

    Returns:
        (N, 3) RGB colors in [0, 1] range.
    """
    distance = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    d_norm = np.clip((distance - d_min) / (d_max - d_min), 0, 1)

    # Viridis-like colormap
    colors = np.zeros((len(distance), 3), dtype=np.float32)
    colors[:, 0] = 0.267 + 0.329 * d_norm + 0.404 * d_norm**2
    colors[:, 1] = 0.004 + 0.873 * d_norm - 0.377 * d_norm**2
    colors[:, 2] = 0.329 + 0.089 * d_norm - 0.418 * d_norm**2

    return np.clip(colors, 0, 1)
