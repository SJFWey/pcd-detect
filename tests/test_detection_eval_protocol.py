from types import SimpleNamespace

import numpy as np
import pytest

from src.commands import eval_cmd
from src.commands.eval_cmd import _resolve_roi_bounds
from src.eval import detection_eval
from src.eval.detection_eval import BoundingBox3D, InstanceMatcher


def _box(semantic_id: int = 10) -> BoundingBox3D:
    return BoundingBox3D(
        center=np.zeros(3, dtype=np.float64),
        dimensions=np.ones(3, dtype=np.float64),
        yaw=0.0,
        semantic_id=semantic_id,
    )


def test_instance_matcher_maximizes_threshold_valid_match_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        detection_eval,
        "compute_iou_matrix",
        lambda *args: np.array([[0.90, 0.51], [0.50, 0.49]], dtype=np.float64),
    )

    matches, unmatched_preds, unmatched_gts = InstanceMatcher(iou_threshold=0.5).match(
        [_box(), _box()], [_box(), _box()]
    )

    assert matches == [(0, 1, 0.51), (1, 0, 0.5)]
    assert unmatched_preds == []
    assert unmatched_gts == []


def test_instance_matcher_keeps_class_mismatched_zero_edge_infeasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        detection_eval,
        "compute_iou_matrix",
        lambda *args: np.array([[0.0]], dtype=np.float64),
    )

    matches, unmatched_preds, unmatched_gts = InstanceMatcher(
        iou_threshold=0.0, match_classes=True
    ).match([_box(10)], [_box(30)])

    assert matches == []
    assert unmatched_preds == [0]
    assert unmatched_gts == [0]


def test_eval_command_fits_gt_from_the_same_roi_support_as_predictions(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    output_root = tmp_path / "output"
    boxes_path = output_root / "boxes" / "00" / "boxes.jsonl"
    boxes_path.parent.mkdir(parents=True)
    boxes_path.write_text("", encoding="utf-8")
    cfg = {
        "dataset": {"root": str(tmp_path / "dataset"), "sequence": "00"},
        "frames": {"start": 0, "end": 1, "step": 1},
        "preprocess": {
            "roi": {
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
                "z_min": 0.0,
                "z_max": 1.0,
            }
        },
        "output": {
            "root": str(output_root),
            "boxes_dir": "boxes",
            "predictions_dir": "predictions",
            "reports_dir": "reports",
        },
        "evaluation": {
            "official": False,
            "detection": True,
            "perf": False,
            "targets": ["car"],
        },
    }
    points, instance_labels, semantic_labels = _roi_fixture()
    frame = SimpleNamespace(
        points=points, inst_label=instance_labels, sem_label=semantic_labels
    )
    dataset = SimpleNamespace(num_frames=1, get_frame=lambda frame_id: frame)
    captured = {}

    def capture_gt(points_arg, instance_arg, semantic_arg, labels_helper):
        captured["points"] = points_arg.copy()
        captured["instance_labels"] = instance_arg.copy()
        captured["semantic_labels"] = semantic_arg.copy()
        return []

    monkeypatch.setattr(eval_cmd, "load_config", lambda config, task: cfg)
    monkeypatch.setattr(eval_cmd, "KITTIDataset", lambda root, sequence: dataset)
    monkeypatch.setattr(eval_cmd, "extract_gt_instances", capture_gt)
    monkeypatch.setattr(eval_cmd, "_generate_evaluation_curves", lambda *args: None)

    eval_cmd.eval_command()

    _assert_roi_masked_arrays(captured, points, instance_labels, semantic_labels)



def _roi_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.01, 0.0, 0.0], [-0.01, 0.0, 0.0]],
        dtype=np.float32,
    )
    return (
        points,
        np.array([101, 102, 103, 104], dtype=np.uint32),
        np.array([10, 10, 10, 10], dtype=np.uint32),
    )


def _assert_roi_masked_arrays(
    captured: dict[str, np.ndarray],
    points: np.ndarray,
    instance_labels: np.ndarray,
    semantic_labels: np.ndarray,
) -> None:
    np.testing.assert_array_equal(captured["points"], points[:2])
    np.testing.assert_array_equal(captured["instance_labels"], instance_labels[:2])
    np.testing.assert_array_equal(captured["semantic_labels"], semantic_labels[:2])


def test_resolve_roi_bounds_uses_pipeline_defaults_for_partial_config() -> None:
    bounds = _resolve_roi_bounds({"preprocess": {"roi": {"x_max": 25.0}}})

    assert bounds.x_min == 0.0
    assert bounds.x_max == 25.0
    assert (bounds.y_min, bounds.y_max, bounds.z_min, bounds.z_max) == (-40.0, 40.0, -3.0, 3.0)
