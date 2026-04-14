from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from app.adapters.base import DatasetAdapter, DatasetMetadata, FrameSample
from app.models import DatasetConfig, DatasetSceneOption, SensorMatch

# Keep the order aligned with KITTI/OpenPCDet label ids so worker predictions
# are not renamed to the wrong class after inference.
RORAD_CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
RORAD_DEFAULT_SCENE_ID = "ped_1"
RORAD_ROOT = Path(os.environ.get("RORAD_DATASET_ROOT", "/workspace-data/rorad"))

DATASET_DEFINITION = {
    "id": "rorad",
    "name": "RoRad",
    "class_names": RORAD_CLASS_NAMES,
    "default_scene_id": RORAD_DEFAULT_SCENE_ID,
}


def _rorad_kitti_root() -> Path:
    return RORAD_ROOT / "kitti"


def _rorad_scene_root(scene_id: str) -> Path:
    return _rorad_kitti_root() / scene_id


def _timestamp_from_stem(stem: str) -> float:
    token = stem.rsplit("_", 1)[-1]
    timestamp_ns = int(token)
    return timestamp_ns / 1_000_000_000.0


def _probe_rorad_scene(scene_id: str) -> tuple[bool, str | None]:
    scene_root = _rorad_scene_root(scene_id)
    lidar_root = scene_root / "lidar"
    camera_root = scene_root / "camera"
    if not scene_root.exists():
        return False, f"{scene_id} is not mounted inside the API container"
    if next(lidar_root.glob("*.bin"), None) is None:
        return False, f"{scene_id} does not contain LiDAR .bin files"
    if next(camera_root.glob("*.png"), None) is None:
        return False, f"{scene_id} does not contain camera .png files"
    return True, None


def list_dataset_scenes() -> list[DatasetSceneOption]:
    kitti_root = _rorad_kitti_root()
    scene_ids = sorted(path.name for path in kitti_root.iterdir() if path.is_dir()) if kitti_root.exists() else []
    if RORAD_DEFAULT_SCENE_ID not in scene_ids:
        scene_ids.insert(0, RORAD_DEFAULT_SCENE_ID)

    scenes = []
    for scene_id in scene_ids:
        available, reason = _probe_rorad_scene(scene_id)
        scenes.append(
            DatasetSceneOption(
                id=scene_id,
                name=scene_id,
                available=available,
                reason=reason,
            )
        )
    return scenes


def build_dataset_config(scene_id: str) -> DatasetConfig:
    return DatasetConfig(
        adapter="rorad",
        dataset_root=str(_rorad_scene_root(scene_id)),
        sequence_name=scene_id,
        class_names=RORAD_CLASS_NAMES,
        camera_tolerance_ms=75.0,
        radar_tolerance_ms=125.0,
    )


class RoRadAdapter(DatasetAdapter):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.lidar_files = sorted((self.dataset_root / "lidar").glob("*.bin"))
        self.camera_files = sorted((self.dataset_root / "camera").glob("*.png"))
        self.camera_by_stem = {path.stem: path for path in self.camera_files}
        self.timeline = [self._build_summary(index, path) for index, path in enumerate(self.lidar_files)]
        self.fps = self._estimate_fps()
        self.calibration = {
            "dataset_root": str(self.dataset_root),
            "note": "RoRad loader currently exposes LiDAR and camera only; radar ADC visualization is intentionally disabled.",
        }

        if not self.timeline:
            raise ValueError(f"No RoRad LiDAR .bin files found in {self.dataset_root / 'lidar'}")

    def __len__(self) -> int:
        return len(self.timeline)

    def _estimate_fps(self) -> float:
        if len(self.lidar_files) < 2:
            return 0.0
        timestamps = np.array([_timestamp_from_stem(path.stem) for path in self.lidar_files], dtype=np.float64)
        deltas = np.diff(timestamps)
        mean_delta = float(np.mean(deltas)) if deltas.size else 0.0
        return 1.0 / mean_delta if mean_delta > 0.0 else 0.0

    def _build_camera_match(self, lidar_path: Path, timestamp: float) -> dict[str, SensorMatch]:
        camera_path = self.camera_by_stem.get(lidar_path.stem)
        return {
            "rgb": SensorMatch(
                index=None,
                timestamp=timestamp if camera_path else None,
                path=str(camera_path) if camera_path else None,
                delta_ms=0.0 if camera_path else None,
                available=camera_path is not None,
            )
        }

    def _build_summary(self, index: int, lidar_path: Path) -> dict:
        timestamp = _timestamp_from_stem(lidar_path.stem)
        return {
            "frame_index": index,
            "timestamp": timestamp,
            "lidar_path": str(lidar_path),
            "camera": self._build_camera_match(lidar_path, timestamp),
            "radar": {},
        }

    def get_frame_summary(self, index: int) -> dict:
        return self.timeline[index]

    def get_frame(self, index: int) -> FrameSample:
        summary = self.timeline[index]
        lidar_points = np.fromfile(summary["lidar_path"], dtype=np.float32)
        if lidar_points.size % 4 != 0:
            raise ValueError(f"RoRad LiDAR file is not float32 x/y/z/intensity: {summary['lidar_path']}")
        lidar_points = lidar_points.reshape(-1, 4)
        return FrameSample(
            frame_index=index,
            timestamp=summary["timestamp"],
            lidar_points=lidar_points,
            lidar_path=summary["lidar_path"],
            camera_frames=summary["camera"],
            radar_frames=summary["radar"],
            calibration=self.calibration,
            ego_pose=None,
        )

    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            sequence_name=self.config.sequence_name,
            class_names=self.config.class_names,
            fps=self.fps,
            calibration=self.calibration,
        )
