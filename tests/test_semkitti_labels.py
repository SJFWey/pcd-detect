import numpy as np

from src.datasets.semkitti_labels import SemanticKITTILabels


def test_project_owned_mapping_colorizes_and_remaps_supported_semantic_ids() -> None:
    labels = SemanticKITTILabels()
    semantic_ids = np.array([10, 50, 252, 259], dtype=np.uint16)

    colors = labels.colorize_labels(semantic_ids)

    assert colors.shape == (4, 3)
    assert colors[2].any()
    assert labels.map_to_training(semantic_ids).tolist() == [1, 13, 1, 5]
    assert labels.map_from_training(np.array([1, 13], dtype=np.int32)).tolist() == [10, 50]
