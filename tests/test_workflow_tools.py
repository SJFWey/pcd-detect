import json
from pathlib import Path

import yaml

from tools import run_ablation, run_all


def test_run_all_stops_after_failed_dataset_validation(
    monkeypatch, tmp_path: Path
) -> None:
    config = {
        "dataset": {"root": str(tmp_path), "sequence": "08"},
        "output": {
            "root": str(tmp_path / "output"),
            "predictions_dir": "predictions",
            "reports_dir": "reports",
            "videos_dir": "videos",
        },
    }
    calls: list[str] = []

    monkeypatch.setattr(run_all, "load_config", lambda **kwargs: config)
    monkeypatch.setattr(
        run_all,
        "run_command",
        lambda command, description: calls.append(description) or False,
    )
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        ["run_all.py", "--skip-export"],
    )

    assert run_all.main() == 1
    assert calls == ["Dataset Validation"]


def test_run_all_stops_before_evaluation_after_failed_detection(
    monkeypatch, tmp_path: Path
) -> None:
    config = {
        "dataset": {"root": str(tmp_path), "sequence": "08"},
        "output": {"root": str(tmp_path / "output")},
    }
    calls: list[str] = []
    outcomes = iter([True, False])

    monkeypatch.setattr(run_all, "load_config", lambda **kwargs: config)
    monkeypatch.setattr(
        run_all,
        "run_command",
        lambda command, description: calls.append(description) or next(outcomes),
    )
    monkeypatch.setattr(
        run_all.sys,
        "argv",
        ["run_all.py", "--skip-export"],
    )

    assert run_all.main() == 1
    assert calls == ["Dataset Validation", "Detection Pipeline"]


def test_ablation_config_has_one_explicit_reproducible_protocol() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "ablation"
        / "proposal_cmp.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["dataset"]["sequence"] == "08"
    assert config["preprocess"]["roi"] == {
        "x_min": 0.0,
        "x_max": 70.0,
        "y_min": -40.0,
        "y_max": 40.0,
        "z_min": -3.0,
        "z_max": 3.0,
    }
    assert config["ground"]["random_seed"] == 0
    assert config["evaluation"]["target_semantic_ids"] == [10, 30]


def test_ablation_report_has_computed_conclusions(tmp_path: Path) -> None:
    result = run_ablation.AblationResult(
        method="dbscan_fixed",
        description="Fixed DBSCAN",
        metrics={
            "precision": 0.1,
            "recall": 0.2,
            "f1": 0.133,
            "mean_iou": 0.6,
            "num_proposals_avg": 4.0,
        },
        distance_metrics={"0-10m": {"precision": 0.1, "recall": 0.2, "f1": 0.133}},
        timing={"total": {"mean_ms": 12.0, "p50_ms": 11.0}},
        num_frames=2,
    )

    run_ablation.generate_report([result], tmp_path)
    report = (tmp_path / "ablation_proposals.md").read_text(encoding="utf-8")

    assert "TODO" not in report
    assert "Highest proxy F1" in report
    assert "not KITTI 3D detection AP" in report


def test_published_metrics_reconcile_overall_and_distance_totals() -> None:
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "results"
        / "kitti08_subset.metrics.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    overall = artifact["overall"]
    bins = artifact["by_distance"].values()

    assert sum(item["predictions"] for item in bins) == overall["predictions"]
    assert sum(item["reference_boxes"] for item in bins) == overall["reference_boxes"]
    assert sum(item["tp"] for item in bins) == overall["true_positives"]
    assert sum(item["fp"] for item in bins) == overall["false_positives"]
    assert sum(item["fn"] for item in bins) == overall["false_negatives"]
