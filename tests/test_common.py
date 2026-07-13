import pytest
import typer

from src.commands.common import parse_frame_range
from src.commands.eval_cmd import _resolve_eval_frame_range


def test_parse_frame_range_for_interval() -> None:
    frame_range = parse_frame_range("3-8", 20)

    assert list(frame_range) == [3, 4, 5, 6, 7]


def test_parse_frame_range_clamps_end_to_num_frames() -> None:
    frame_range = parse_frame_range("8-20", 10)

    assert list(frame_range) == [8, 9]


def test_parse_frame_range_rejects_empty_input() -> None:
    with pytest.raises(typer.BadParameter):
        parse_frame_range(" ", 10)


def test_resolve_eval_frame_range_uses_configured_subset() -> None:
    cfg = {"frames": {"start": 5, "end": 12, "step": 2}}

    frame_range = _resolve_eval_frame_range(cfg, 100)

    assert list(frame_range) == [5, 7, 9, 11]
