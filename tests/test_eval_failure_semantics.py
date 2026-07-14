import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
import yaml

from src.commands import eval_cmd
from src.datasets.semkitti_labels import SemanticKITTILabels


REPO_ROOT = Path(__file__).resolve().parents[1]
MINI_CONFIG = REPO_ROOT / "examples/mini_config.yaml"


def run_eval(config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src.main", "eval", "--config", str(config)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def write_mini_config(tmp_path: Path, output_root: Path) -> Path:
    config = yaml.safe_load(MINI_CONFIG.read_text(encoding="utf-8"))
    config["output"]["root"] = str(output_root)
    config_path = tmp_path / "mini_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_project_owned_label_mapping_supports_mini_fixture_labels() -> None:
    labels = SemanticKITTILabels()

    assert "third_party" not in str(labels.config_path)
    assert labels.get_label_name(10) == "car"
    assert labels.get_label_name(30) == "person"
    assert labels.get_label_name(40) == "road"


def test_mini_detection_eval_writes_parseable_metrics_from_project_owned_mapping() -> None:
    output_root = REPO_ROOT / "outputs/mini"
    shutil.rmtree(output_root, ignore_errors=True)

    run_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.main",
            "run",
            "--config",
            "examples/mini_config.yaml",
            "--frames",
            "0-2",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr

    eval_result = run_eval(MINI_CONFIG)
    report_path = output_root / "reports/det_metrics.json"
    perf_path = output_root / "reports/perf.json"

    assert eval_result.returncode == 0, eval_result.stderr
    assert "Evaluation complete!" in eval_result.stdout
    assert json.loads(report_path.read_text(encoding="utf-8"))
    assert json.loads(perf_path.read_text(encoding="utf-8"))


def test_detection_eval_missing_label_mapping_exits_nonzero_without_completion_banner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = {
        "dataset": {"root": str(tmp_path / "dataset"), "sequence": "00"},
        "output": {
            "root": str(tmp_path / "output"),
            "predictions_dir": "predictions",
            "reports_dir": "reports",
        },
        "evaluation": {"official": False, "detection": True, "perf": False},
    }
    monkeypatch.setattr(eval_cmd, "load_config", lambda config, task: cfg)
    monkeypatch.setattr(eval_cmd, "KITTIDataset", lambda root, sequence: object())

    def raise_missing_mapping() -> None:
        raise FileNotFoundError("project label mapping is missing")

    monkeypatch.setattr(eval_cmd, "SemanticKITTILabels", raise_missing_mapping)

    with pytest.raises(typer.Exit) as exc_info:
        eval_cmd.eval_command()

    assert exc_info.value.exit_code == 1
    assert "Evaluation complete!" not in capsys.readouterr().out


def test_detection_eval_missing_boxes_exits_nonzero_without_completion_banner(
    tmp_path: Path,
) -> None:
    config_path = write_mini_config(tmp_path, tmp_path / "missing-output")

    result = run_eval(config_path)

    assert result.returncode != 0
    assert "Evaluation complete!" not in result.stdout


def test_detection_eval_malformed_boxes_exits_nonzero_without_completion_banner(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    boxes_path = output_root / "boxes/00/boxes.jsonl"
    boxes_path.parent.mkdir(parents=True)
    boxes_path.write_text("not valid json\n", encoding="utf-8")
    config_path = write_mini_config(tmp_path, output_root)

    result = run_eval(config_path)

    assert result.returncode != 0
    assert "Evaluation complete!" not in result.stdout


def test_official_evaluator_nonzero_exit_is_propagated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    predictions = tmp_path / "predictions"
    (predictions / "sequences").mkdir(parents=True)
    output_root = tmp_path / "output"
    cfg = {
        "dataset": {"root": str(tmp_path / "dataset"), "sequence": "00"},
        "output": {
            "root": str(output_root),
            "predictions_dir": "predictions",
            "reports_dir": "reports",
        },
        "evaluation": {
            "official": True,
            "detection": False,
            "perf": False,
            "split": "valid",
            "predictions_path": str(predictions),
        },
    }
    monkeypatch.setattr(eval_cmd, "load_config", lambda config, task: cfg)
    monkeypatch.setattr(
        eval_cmd.subprocess,
        "run",
        lambda command: subprocess.CompletedProcess(command, returncode=9),
    )

    with pytest.raises(typer.Exit) as exc_info:
        eval_cmd.eval_command()

    assert exc_info.value.exit_code == 9
    assert "Evaluation complete!" not in capsys.readouterr().out


def test_missing_official_evaluator_wrapper_exits_without_completion_banner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    predictions = tmp_path / "predictions"
    (predictions / "sequences").mkdir(parents=True)
    cfg = {
        "dataset": {"root": str(tmp_path / "dataset"), "sequence": "00"},
        "output": {
            "root": str(tmp_path / "output"),
            "predictions_dir": "predictions",
            "reports_dir": "reports",
        },
        "evaluation": {
            "official": True,
            "detection": False,
            "perf": False,
            "predictions_path": str(predictions),
        },
    }
    original_exists = eval_cmd.Path.exists

    def exists_without_wrapper(path: Path) -> bool:
        if path.name == "run_official_eval.py":
            return False
        return original_exists(path)

    monkeypatch.setattr(eval_cmd, "load_config", lambda config, task: cfg)
    monkeypatch.setattr(eval_cmd.Path, "exists", exists_without_wrapper)

    with pytest.raises(typer.Exit) as exc_info:
        eval_cmd.eval_command()

    assert exc_info.value.exit_code == 1
    assert "Evaluation complete!" not in capsys.readouterr().out


def test_viz_requested_processing_failure_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src.commands import viz_cmd

    cfg = {
        "dataset": {"root": str(tmp_path / "dataset"), "sequence": "00"},
        "visualization": {"frame": 0, "mode": "boxes"},
    }
    monkeypatch.setattr(viz_cmd, "load_config", lambda config, task: cfg)
    dataset = type("Dataset", (), {"num_frames": 1})()
    monkeypatch.setattr(viz_cmd, "KITTIDataset", lambda root, sequence: dataset)
    monkeypatch.setattr(
        viz_cmd,
        "_process_single_frame_for_viz",
        lambda config, dataset, frame: (_ for _ in ()).throw(ValueError("bad frame")),
    )

    with pytest.raises(typer.Exit) as exc_info:
        viz_cmd.viz_command(output=tmp_path / "frame.png")

    assert exc_info.value.exit_code == 1


def test_export_missing_requested_boxes_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src.commands import export_cmd

    cfg = {
        "dataset": {"root": str(tmp_path / "dataset"), "sequence": "00"},
        "output": {
            "root": str(tmp_path / "output"),
            "boxes_dir": "boxes",
            "predictions_dir": "predictions",
            "reports_dir": "reports",
            "videos_dir": "videos",
        },
        "export": {"type": "predictions", "from_boxes": True, "start": 0, "end": 1},
        "video": {"sequence": {"step": 1}},
    }
    monkeypatch.setattr(export_cmd, "load_config", lambda config, task: cfg)

    with pytest.raises(typer.BadParameter):
        export_cmd.export_command()


def test_ground_truth_export_partial_failure_exits_without_completion_banner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.commands import export_cmd

    dataset = SimpleNamespace(has_labels=True, num_frames=1)
    stats = SimpleNamespace(
        num_exported=0,
        num_frames=1,
        total_points=0,
        num_failed=1,
        failed_frames=[0],
    )

    class FakeExporter:
        predictions_dir = tmp_path / "predictions"

        def __init__(self, **kwargs) -> None:
            pass

        def export_ground_truth_as_predictions(self, **kwargs):
            return stats

    monkeypatch.setattr(export_cmd, "KITTIDataset", lambda root, sequence: dataset)
    monkeypatch.setattr(export_cmd, "PredictionExporter", FakeExporter)

    with pytest.raises(typer.Exit) as exc_info:
        export_cmd._export_predictions(
            {"dataset": {"root": str(tmp_path / "dataset"), "sequence": "00"}},
            tmp_path / "predictions",
            0,
            1,
        )

    assert exc_info.value.exit_code == 1
    assert "Export complete!" not in capsys.readouterr().out


def test_detection_eval_with_no_labeled_frames_exits_without_completion_banner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_root = tmp_path / "output"
    boxes_path = output_root / "boxes/00/boxes.jsonl"
    boxes_path.parent.mkdir(parents=True)
    boxes_path.write_text("", encoding="utf-8")
    cfg = {
        "dataset": {"root": str(tmp_path / "dataset"), "sequence": "00"},
        "output": {
            "root": str(output_root),
            "boxes_dir": "boxes",
            "predictions_dir": "predictions",
            "reports_dir": "reports",
        },
        "evaluation": {"official": False, "detection": True, "perf": False},
    }
    frame = SimpleNamespace(inst_label=None, sem_label=None)
    dataset = SimpleNamespace(num_frames=1, get_frame=lambda frame_id: frame)
    monkeypatch.setattr(eval_cmd, "load_config", lambda config, task: cfg)
    monkeypatch.setattr(eval_cmd, "KITTIDataset", lambda root, sequence: dataset)

    with pytest.raises(typer.BadParameter):
        eval_cmd.eval_command()

    assert "Evaluation complete!" not in capsys.readouterr().out
