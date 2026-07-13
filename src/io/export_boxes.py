"""Export bounding box results in structured formats.

Supports JSONL and Parquet output formats for per-frame box results.

JSONL Format (one JSON object per line):
{
    "frame_id": 0,
    "timestamp": "2024-01-01T00:00:00",
    "num_boxes": 5,
    "boxes": [
        {
            "box_id": 0,
            "center": [x, y, z],
            "extent": [ex, ey, ez],
            "yaw": 0.5,
            "dimensions": [length, width, height],
            "volume": 10.5,
            "num_points": 100,
            "score": 0.95
        },
        ...
    ]
}
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from ..boxes.obb import OBBFitResult


@dataclass(frozen=True)
class BoxRecord:
    """Single bounding box record for export."""

    box_id: int
    center: tuple[float, float, float]
    extent: tuple[float, float, float]
    yaw: float
    dimensions: tuple[float, float, float]
    volume: float
    num_points: int
    score: float = 1.0
    semantic_id: int = 0
    valid: bool = True
    cluster_id: int = -1

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "box_id": self.box_id,
            "center": list(self.center),
            "extent": list(self.extent),
            "yaw": self.yaw,
            "dimensions": list(self.dimensions),
            "volume": self.volume,
            "num_points": self.num_points,
            "score": self.score,
            "semantic_id": self.semantic_id,
            "valid": self.valid,
            "cluster_id": self.cluster_id,
        }


@dataclass(frozen=True)
class FrameBoxes:
    """All bounding boxes for a single frame."""

    frame_id: int
    boxes: list[BoxRecord]
    timestamp: str | None = None
    processing_time_ms: float | None = None

    @property
    def num_boxes(self) -> int:
        return len(self.boxes)

    @property
    def valid_boxes(self) -> list[BoxRecord]:
        return [b for b in self.boxes if b.valid]

    @property
    def num_valid_boxes(self) -> int:
        return len(self.valid_boxes)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result: dict = {
            "frame_id": self.frame_id,
            "num_boxes": self.num_boxes,
            "num_valid_boxes": self.num_valid_boxes,
            "boxes": [b.to_dict() for b in self.boxes],
        }
        if self.timestamp:
            result["timestamp"] = self.timestamp
        if self.processing_time_ms is not None:
            result["processing_time_ms"] = self.processing_time_ms
        return result


def obb_results_to_box_records(
    obb_results: Sequence[OBBFitResult],
    scores: Sequence[float] | None = None,
) -> list[BoxRecord]:
    """
    Convert OBB fit results to BoxRecord list.

    Args:
        obb_results: List of OBB fitting results.
        scores: Optional confidence scores for each box.

    Returns:
        List of BoxRecord objects.
    """
    if scores is None:
        scores = [1.0] * len(obb_results)
    elif len(scores) != len(obb_results):
        raise ValueError(
            f"scores length ({len(scores)}) must match obb_results length ({len(obb_results)})"
        )

    records = []
    box_id = 0

    for result, score in zip(obb_results, scores):
        if result.params is None:
            # Include invalid boxes with zero dimensions
            record = BoxRecord(
                box_id=box_id,
                center=(0.0, 0.0, 0.0),
                extent=(0.0, 0.0, 0.0),
                yaw=0.0,
                dimensions=(0.0, 0.0, 0.0),
                volume=0.0,
                num_points=result.num_points,
                score=score,
                semantic_id=0,
                valid=False,
                cluster_id=result.cluster_id,
            )
        else:
            params = result.params
            record = BoxRecord(
                box_id=box_id,
                center=tuple(float(x) for x in params.center),  # type: ignore
                extent=tuple(float(x) for x in params.extent),  # type: ignore
                yaw=params.yaw,
                dimensions=params.dimensions,
                volume=params.volume,
                num_points=result.num_points,
                score=score,
                semantic_id=0,
                valid=result.valid,
                cluster_id=result.cluster_id,
            )
        records.append(record)
        box_id += 1

    return records


def create_frame_boxes(
    frame_id: int,
    obb_results: Sequence[OBBFitResult],
    scores: Sequence[float] | None = None,
    processing_time_ms: float | None = None,
) -> FrameBoxes:
    """
    Create FrameBoxes from OBB fit results.

    Args:
        frame_id: Frame identifier.
        obb_results: List of OBB fitting results.
        scores: Optional confidence scores for each box.
        processing_time_ms: Optional processing time in milliseconds.

    Returns:
        FrameBoxes object.
    """
    records = obb_results_to_box_records(obb_results, scores)
    timestamp = datetime.now().isoformat()

    return FrameBoxes(
        frame_id=frame_id,
        boxes=records,
        timestamp=timestamp,
        processing_time_ms=processing_time_ms,
    )


class BoxesExporter:
    """
    Exporter for bounding box results.

    Supports JSONL and Parquet formats.
    """

    def __init__(
        self,
        output_dir: str | Path,
        sequence: str,
        boxes_dir: str = "boxes",
        format: str = "jsonl",
    ):
        """
        Initialize exporter.

        Args:
            output_dir: Output directory root.
            sequence: Sequence identifier (e.g., "00").
            boxes_dir: Subdirectory under output_dir for box files.
            format: Output format ("jsonl" or "parquet").
        """
        self.output_dir = Path(output_dir)
        self.sequence = str(sequence).zfill(2)
        self.boxes_dir_name = boxes_dir
        self.format = format.lower()

        if self.format not in ("jsonl", "parquet"):
            raise ValueError(f"Unsupported format: {self.format}")

        # Create output directory
        self.boxes_dir = self.output_dir / self.boxes_dir_name / self.sequence
        self.boxes_dir.mkdir(parents=True, exist_ok=True)

        # Initialize output file
        self._jsonl_path = self.boxes_dir / "boxes.jsonl"
        self._parquet_path = self.boxes_dir / "boxes.parquet"
        self._frame_records: list[dict] = []
        if self.format == "jsonl" and self._jsonl_path.exists():
            self._jsonl_path.unlink()

    def export_frame(self, frame_boxes: FrameBoxes) -> None:
        """
        Export a single frame's boxes.

        For JSONL: appends to file immediately.
        For Parquet: accumulates records for batch export.
        """
        record = frame_boxes.to_dict()

        if self.format == "jsonl":
            self._write_jsonl_record(record)
        else:
            self._frame_records.append(record)

    def _write_jsonl_record(self, record: dict) -> None:
        """Write a single record to JSONL file."""
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finalize(self) -> Path:
        """
        Finalize export and return output path.

        For Parquet: writes accumulated records to file.
        For JSONL: returns the file path.
        """
        if self.format == "parquet":
            return self._write_parquet()
        return self._jsonl_path

    def _write_parquet(self) -> Path:
        """Write accumulated records to Parquet file."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet export. "
                "Install with: uv add pyarrow"
            )

        # Flatten records for columnar storage
        flat_records = []
        for frame_record in self._frame_records:
            frame_id = frame_record["frame_id"]
            timestamp = frame_record.get("timestamp", "")
            processing_time_ms = frame_record.get("processing_time_ms", 0.0)

            for box in frame_record["boxes"]:
                flat_record = {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "processing_time_ms": processing_time_ms,
                    "box_id": box["box_id"],
                    "center_x": box["center"][0],
                    "center_y": box["center"][1],
                    "center_z": box["center"][2],
                    "extent_x": box["extent"][0],
                    "extent_y": box["extent"][1],
                    "extent_z": box["extent"][2],
                    "yaw": box["yaw"],
                    "length": box["dimensions"][0],
                    "width": box["dimensions"][1],
                    "height": box["dimensions"][2],
                    "volume": box["volume"],
                    "num_points": box["num_points"],
                    "score": box["score"],
                    "semantic_id": box.get("semantic_id", 0),
                    "valid": box["valid"],
                    "cluster_id": box["cluster_id"],
                }
                flat_records.append(flat_record)

        if not flat_records:
            # Create empty table with schema
            schema = pa.schema(
                [
                    ("frame_id", pa.int32()),
                    ("timestamp", pa.string()),
                    ("processing_time_ms", pa.float64()),
                    ("box_id", pa.int32()),
                    ("center_x", pa.float64()),
                    ("center_y", pa.float64()),
                    ("center_z", pa.float64()),
                    ("extent_x", pa.float64()),
                    ("extent_y", pa.float64()),
                    ("extent_z", pa.float64()),
                    ("yaw", pa.float64()),
                    ("length", pa.float64()),
                    ("width", pa.float64()),
                    ("height", pa.float64()),
                    ("volume", pa.float64()),
                    ("num_points", pa.int32()),
                    ("score", pa.float64()),
                    ("semantic_id", pa.int32()),
                    ("valid", pa.bool_()),
                    ("cluster_id", pa.int32()),
                ]
            )
            table = pa.Table.from_pydict({}, schema=schema)
        else:
            table = pa.Table.from_pylist(flat_records)

        pq.write_table(table, self._parquet_path)
        return self._parquet_path

    def clear(self) -> None:
        """Clear accumulated records and reset output files."""
        self._frame_records = []
        if self._jsonl_path.exists():
            self._jsonl_path.unlink()


def export_boxes_jsonl(
    output_path: str | Path,
    frame_boxes_list: Sequence[FrameBoxes],
) -> Path:
    """
    Export multiple frames' boxes to a single JSONL file.

    Args:
        output_path: Output file path.
        frame_boxes_list: List of FrameBoxes objects.

    Returns:
        Path to the output file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for frame_boxes in frame_boxes_list:
            f.write(json.dumps(frame_boxes.to_dict(), ensure_ascii=False) + "\n")

    return output_path


def load_boxes_jsonl(input_path: str | Path) -> list[FrameBoxes]:
    """
    Load boxes from JSONL file.

    Args:
        input_path: Input file path.

    Returns:
        List of FrameBoxes objects.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    frame_boxes_list = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            boxes = [
                BoxRecord(
                    box_id=b["box_id"],
                    center=tuple(b["center"]),  # type: ignore
                    extent=tuple(b["extent"]),  # type: ignore
                    yaw=b["yaw"],
                    dimensions=tuple(b["dimensions"]),  # type: ignore
                    volume=b["volume"],
                    num_points=b["num_points"],
                    score=b.get("score", 1.0),
                    semantic_id=b.get("semantic_id", 0),
                    valid=b.get("valid", True),
                    cluster_id=b.get("cluster_id", -1),
                )
                for b in record["boxes"]
            ]
            frame_boxes = FrameBoxes(
                frame_id=record["frame_id"],
                boxes=boxes,
                timestamp=record.get("timestamp"),
                processing_time_ms=record.get("processing_time_ms"),
            )
            frame_boxes_list.append(frame_boxes)

    return frame_boxes_list


class JSONLBoxReader:
    """Reader for JSONL box files with random access by frame ID.

    Loads the entire file into memory and indexes by frame ID for
    efficient lookups during evaluation.

    Usage:
        reader = JSONLBoxReader("outputs/boxes_00.jsonl")
        frame_boxes = reader.read_frame(42)
        if frame_boxes:
            for box in frame_boxes.boxes:
                print(box.center)
    """

    def __init__(self, input_path: str | Path):
        """Initialize reader and load data.

        Args:
            input_path: Path to JSONL file.
        """
        self.input_path = Path(input_path)
        self._frame_index: dict[int, FrameBoxes] = {}
        self._load()

    def _load(self) -> None:
        """Load and index all frames."""
        if not self.input_path.exists():
            return

        frames = load_boxes_jsonl(self.input_path)
        for frame in frames:
            self._frame_index[frame.frame_id] = frame

    def read_frame(self, frame_id: int) -> FrameBoxes | None:
        """Get boxes for a specific frame.

        Args:
            frame_id: Frame index.

        Returns:
            FrameBoxes or None if frame not found.
        """
        return self._frame_index.get(frame_id)

    def get_all_frames(self) -> list[FrameBoxes]:
        """Get all loaded frames sorted by frame ID."""
        return sorted(self._frame_index.values(), key=lambda f: f.frame_id)

    @property
    def frame_ids(self) -> list[int]:
        """Get list of available frame IDs."""
        return sorted(self._frame_index.keys())

    @property
    def num_frames(self) -> int:
        """Get number of loaded frames."""
        return len(self._frame_index)

    def __len__(self) -> int:
        """Return number of frames."""
        return self.num_frames

    def __getitem__(self, frame_id: int) -> FrameBoxes:
        """Get frame by ID, raises KeyError if not found."""
        if frame_id not in self._frame_index:
            raise KeyError(f"Frame {frame_id} not found")
        return self._frame_index[frame_id]

    def __contains__(self, frame_id: int) -> bool:
        """Check if frame ID is available."""
        return frame_id in self._frame_index
