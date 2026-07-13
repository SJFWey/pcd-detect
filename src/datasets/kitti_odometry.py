"""src.datasets.kitti_odometry

Lightweight KITTI Odometry (+ optional SemanticKITTI labels) reader.

Design goals:
- Read point clouds via NumPy (`np.fromfile`) for maximum simplicity/speed.
- Parse SemanticKITTI `.label` strictly as uint32 per point and split
    semantic/instance ids using the official bit layout.
- Reuse the official `semantic-kitti.yaml` (vendored in third_party) for
    label definitions and learning_map / learning_map_inv, without importing
    the SemanticKITTI API as the primary data loader.
"""

from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..utils.logging import get_logger
from .semkitti_labels import read_label_file

logger = get_logger(__name__)


class FrameData(NamedTuple):
    """Data container for a single frame."""

    frame_id: int
    points: NDArray[np.float32]  # (N, 3) xyz coordinates
    remissions: NDArray[np.float32]  # (N,) remission/intensity values
    sem_label: NDArray[np.uint32] | None  # (N,) semantic ids (lower 16 bits)
    inst_label: NDArray[np.uint32] | None  # (N,) instance ids (upper 16 bits)


class KITTIDataset:
    """
    KITTI Odometry dataset reader with optional SemanticKITTI label support.

    Loads point clouds and labels using NumPy file readers.
    Expects the standard SemanticKITTI directory structure:
        {root}/sequences/{sequence}/velodyne/*.bin
        {root}/sequences/{sequence}/labels/*.label (optional)
        {root}/sequences/{sequence}/calib.txt
        {root}/sequences/{sequence}/poses.txt (sequences 00-10 only)

    Attributes:
        root: Path to dataset root directory.
        sequence: Sequence ID (e.g., "00", "01", ..., "21").
        has_labels: Whether semantic labels are available.
        num_frames: Total number of frames in the sequence.
    """

    def __init__(
        self,
        root: str | Path,
        sequence: str = "00",
    ):
        """
        Initialize KITTI dataset reader.

        Args:
            root: Path to dataset root directory.
            sequence: Sequence ID (e.g., "00").
        """
        self.root = Path(root)
        self.sequence = sequence.zfill(2)  # Ensure 2-digit format

        # Build paths
        self.sequence_path = self.root / "sequences" / self.sequence
        self.velodyne_path = self.sequence_path / "velodyne"
        self.label_path = self.sequence_path / "labels"

        # Validate paths
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        if not self.sequence_path.exists():
            raise FileNotFoundError(f"Sequence not found: {self.sequence_path}")
        if not self.velodyne_path.exists():
            raise FileNotFoundError(f"Velodyne path not found: {self.velodyne_path}")

        # Get list of scan files
        self._scan_files = sorted(self.velodyne_path.glob("*.bin"))
        if not self._scan_files:
            raise FileNotFoundError(f"No .bin files found in {self.velodyne_path}")

        # Check for labels
        self.has_labels = self.label_path.exists() and any(
            self.label_path.glob("*.label")
        )

        logger.info(
            f"Initialized KITTI dataset: sequence={self.sequence}, "
            f"frames={self.num_frames}, has_labels={self.has_labels}"
        )

    @property
    def num_frames(self) -> int:
        """Return total number of frames in the sequence."""
        return len(self._scan_files)

    def __len__(self) -> int:
        """Return total number of frames."""
        return self.num_frames

    @staticmethod
    def _read_scan_file(
        scan_path: Path,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Read a KITTI `.bin` scan into (points, remissions)."""
        if not scan_path.exists():
            raise FileNotFoundError(f"Scan file not found: {scan_path}")

        file_size = scan_path.stat().st_size
        if file_size % 16 != 0:
            raise ValueError(
                f"Invalid .bin size {file_size} (expected multiple of 16 bytes): {scan_path}"
            )

        scan = np.fromfile(str(scan_path), dtype=np.float32)
        if scan.size % 4 != 0:
            raise ValueError(
                f"Invalid .bin float count {scan.size} (expected multiple of 4): {scan_path}"
            )

        scan = scan.reshape(-1, 4)
        points = scan[:, :3]
        remissions = scan[:, 3]
        return points, remissions

    @staticmethod
    def _read_label_file_split(
        label_path: Path,
    ) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
        """Read a SemanticKITTI `.label` and split into (semantic, instance) ids."""
        labels = read_label_file(label_path)
        sem = labels & np.uint32(0xFFFF)
        inst = labels >> np.uint32(16)
        return sem.astype(np.uint32), inst.astype(np.uint32)

    def _get_scan_path(self, frame_id: int) -> Path:
        """Get scan file path for a frame."""
        return self.velodyne_path / f"{frame_id:06d}.bin"

    def _get_label_path(self, frame_id: int) -> Path:
        """Get label file path for a frame."""
        return self.label_path / f"{frame_id:06d}.label"

    def get_frame(self, frame_id: int) -> FrameData:
        """
        Load a single frame's point cloud and labels.

        Args:
            frame_id: Frame index (0-based).

        Returns:
            FrameData containing points, remissions, and optional labels.

        Raises:
            IndexError: If frame_id is out of range.
            FileNotFoundError: If scan file is missing.
        """
        if frame_id < 0 or frame_id >= self.num_frames:
            raise IndexError(
                f"Frame {frame_id} out of range [0, {self.num_frames - 1}]"
            )

        scan_path = self._get_scan_path(frame_id)
        points, remissions = self._read_scan_file(scan_path)

        sem_label: NDArray[np.uint32] | None = None
        inst_label: NDArray[np.uint32] | None = None

        if self.has_labels:
            label_path = self._get_label_path(frame_id)
            if label_path.exists():
                sem_label, inst_label = self._read_label_file_split(label_path)
                if sem_label.shape[0] != points.shape[0]:
                    raise ValueError(
                        "Point/label count mismatch for "
                        f"frame {frame_id:06d}: {points.shape[0]} points vs "
                        f"{sem_label.shape[0]} labels"
                    )

        return FrameData(
            frame_id=frame_id,
            points=points,
            remissions=remissions,
            sem_label=sem_label,
            inst_label=inst_label,
        )

    def __getitem__(self, frame_id: int) -> FrameData:
        """Get frame by index."""
        return self.get_frame(frame_id)

    def __iter__(self) -> Iterator[FrameData]:
        """Iterate over all frames."""
        for i in range(self.num_frames):
            yield self.get_frame(i)

    def get_frame_info(self, frame_id: int) -> dict:
        """
        Get metadata about a frame without loading full data.

        Args:
            frame_id: Frame index.

        Returns:
            Dictionary with frame metadata (paths, file sizes, etc.).
        """
        scan_path = self._get_scan_path(frame_id)
        label_path = self._get_label_path(frame_id)

        info = {
            "frame_id": frame_id,
            "scan_path": str(scan_path),
            "scan_exists": scan_path.exists(),
        }

        if scan_path.exists():
            file_size = scan_path.stat().st_size
            # Each point is 4 float32 values (x, y, z, remission) = 16 bytes
            info["scan_size_bytes"] = file_size
            info["expected_points"] = file_size // 16

        if self.has_labels:
            info["label_path"] = str(label_path)
            label_exists = label_path.exists()
            info["label_exists"] = label_exists
            if label_exists:
                label_size = label_path.stat().st_size
                info["label_size_bytes"] = label_size
                # Each label is 1 uint32 = 4 bytes
                info["expected_labels"] = label_size // 4

        return info

    def get_poses(self) -> NDArray[np.float64] | None:
        """
        Load ground truth poses for the sequence.

        Returns:
            (N, 4, 4) array of transformation matrices, or None if not available.
        """
        poses_path = self.sequence_path / "poses.txt"
        if not poses_path.exists():
            logger.warning(f"Poses file not found: {poses_path}")
            return None

        poses = []
        with open(poses_path, "r", encoding="utf-8") as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                if len(values) == 12:
                    pose = np.eye(4)
                    pose[:3, :4] = np.array(values).reshape(3, 4)
                    poses.append(pose)

        return np.array(poses) if poses else None

    def get_calibration(self) -> dict | None:
        """
        Load calibration data for the sequence.

        Returns:
            Dictionary with calibration matrices, or None if not available.
        """
        calib_path = self.sequence_path / "calib.txt"
        if not calib_path.exists():
            logger.warning(f"Calibration file not found: {calib_path}")
            return None

        calib = {}
        with open(calib_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    values = [float(x) for x in value.strip().split()]
                    if key.startswith("P"):
                        calib[key] = np.array(values).reshape(3, 4)
                    elif key == "Tr":
                        calib["Tr"] = np.array(values).reshape(3, 4)

        return calib if calib else None
