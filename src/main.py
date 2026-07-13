"""
Main CLI entry point for pcd-detect.

Provides subcommands: run, eval, viz, export

CLI Design Principle: Minimal CLI options, configuration via YAML files.
Only --config and workflow control flags (--dry-run) are CLI options.
All algorithm parameters are configured in configs/*.yaml files.
"""

from pathlib import Path

import typer

from . import __version__
from .commands.eval_cmd import eval_command
from .commands.export_cmd import export_command
from .commands.run_cmd import run_command
from .commands.viz_cmd import viz_command

app = typer.Typer(
    name="pcd-detect",
    help="3D Point Cloud Detection on KITTI Dataset",
    add_completion=False,
    invoke_without_command=True,
)


@app.command()
def run(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file (default: use configs/*.yaml)",
    ),
    sequence: str | None = typer.Option(
        None,
        "--sequence",
        "-s",
        help="Override dataset sequence (e.g. 00, 04)",
    ),
    frames: str | None = typer.Option(
        None,
        "--frames",
        "-f",
        help="Frame range (e.g. 0-100 or 50)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print execution plan without running",
    ),
) -> None:
    """
    Run the full detection pipeline.

    All parameters are configured via configs/base.yaml and configs/detection.yaml.
    Use --config to override with a custom config file.
    """
    run_command(config=config, sequence=sequence, dry_run=dry_run, frames=frames)


@app.command()
def eval(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file (default: use configs/*.yaml)",
    ),
) -> None:
    """
    Evaluate detection results.

    All parameters are configured via configs/evaluation.yaml.
    Use --config to override with a custom config file.
    """
    eval_command(config=config)


@app.command()
def viz(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file (default: use configs/*.yaml)",
    ),
    frame: int | None = typer.Option(
        None,
        "--frame",
        help="Override visualization frame index",
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="Override visualization mode: points, boxes, ground, or clusters",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Save a static PNG to this path (disables interactive viewer)",
    ),
) -> None:
    """
    Visualize point clouds and detection results.

    All parameters are configured via configs/visualization.yaml.
    Use --config to override with a custom config file.
    """
    viz_command(config=config, frame=frame, mode=mode, output=output)


@app.command()
def export(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file (default: use configs/*.yaml)",
    ),
) -> None:
    """
    Export results in various formats.

    All parameters are configured via configs/export.yaml.
    Use --config to override with a custom config file.
    """
    export_command(config=config)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
    ),
) -> None:
    """
    pcd-detect: 3D Point Cloud Detection on KITTI Dataset

    A complete pipeline for detecting objects in KITTI Odometry
    point clouds using SemanticKITTI labels for evaluation.
    """
    if version:
        typer.echo(f"pcd-detect version {__version__}")
        raise typer.Exit()


def cli() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    cli()
