from src.io.export_boxes import (
    BoxRecord,
    FrameBoxes,
    JSONLBoxReader,
    export_boxes_jsonl,
    load_boxes_jsonl,
)


def test_jsonl_box_export_round_trip(tmp_path) -> None:
    frame = FrameBoxes(
        frame_id=7,
        boxes=[
            BoxRecord(
                box_id=0,
                center=(1.0, 2.0, 3.0),
                extent=(0.5, 0.5, 1.0),
                yaw=0.25,
                dimensions=(1.0, 1.0, 2.0),
                volume=2.0,
                num_points=42,
                semantic_id=10,
                valid=True,
                cluster_id=3,
            )
        ],
    )
    output_path = tmp_path / "boxes.jsonl"

    export_boxes_jsonl(output_path, [frame])
    loaded = load_boxes_jsonl(output_path)
    reader = JSONLBoxReader(output_path)

    assert len(loaded) == 1
    assert loaded[0].frame_id == 7
    assert loaded[0].boxes[0].center == (1.0, 2.0, 3.0)
    assert reader.frame_ids == [7]
    assert reader.read_frame(7) == loaded[0]
