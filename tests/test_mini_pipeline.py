import subprocess
import sys
from pathlib import Path


def test_mini_detection_pipeline_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.main",
            "run",
            "--config",
            "examples/mini_config.yaml",
            "--frames",
            "0-1",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Pipeline Complete!" in result.stdout
    assert (repo_root / "outputs/mini/boxes/00/boxes.jsonl").exists()
