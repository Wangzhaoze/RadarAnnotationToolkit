#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/datasets/rorad/dataset.py
# @IDE     : vscode

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
RORAD_RAMAP_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
RORAD_CAMERA_INTRINSICS = np.asarray(
    [
        [1152.0, 0.0, 1152.0],
        [0.0, 1152.0, 648.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
RORAD_CAMERA_PROJECTION = np.asarray(
    [
        [1152.0, 0.0, 1152.0, 0.0],
        [0.0, 1152.0, 648.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
RORAD_LIDAR_ORIGIN_IN_CAMERA_FLU = np.asarray([0.1, 0.05, 0.17], dtype=np.float32)
RORAD_CAMERA_FLU_TO_IMAGE_CAMERA = np.asarray(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def _build_rorad_lidar_to_camera_projection_transform() -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = RORAD_CAMERA_FLU_TO_IMAGE_CAMERA
    transform[:3, 3] = RORAD_CAMERA_FLU_TO_IMAGE_CAMERA @ RORAD_LIDAR_ORIGIN_IN_CAMERA_FLU
    return transform


RORAD_LIDAR_TO_CAMERA_PROJECTION = _build_rorad_lidar_to_camera_projection_transform()

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


def _timestamp_from_stem_or_none(stem: str) -> float | None:
    try:
        return _timestamp_from_stem(stem)
    except (TypeError, ValueError):
        return None


def _image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in RORAD_RAMAP_EXTENSIONS)


def _probe_rorad_scene(scene_id: str) -> tuple[bool, str | None]:
    scene_root = _rorad_scene_root(scene_id)
    lidar_root = scene_root / "lidar"
    camera_root = scene_root / "camera"
    ramap_root = scene_root / "ramaps"
    if not scene_root.exists():
        return False, f"{scene_id} is not mounted inside the API container"
    if next(lidar_root.glob("*.bin"), None) is None:
        return False, f"{scene_id} does not contain LiDAR .bin files"
    if next(camera_root.glob("*.png"), None) is None:
        return False, f"{scene_id} does not contain camera .png files"
    if not _image_files(ramap_root):
        return False, f"{scene_id} does not contain RAMap image files in ramaps"
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
        self.radar_files = _image_files(self.dataset_root / "ramaps")
        self.camera_by_stem = {path.stem: path for path in self.camera_files}
        self.camera_index_by_path = {path: index for index, path in enumerate(self.camera_files)}
        self.radar_by_stem = {path.stem: path for path in self.radar_files}
        self.radar_index_by_path = {path: index for index, path in enumerate(self.radar_files)}
        self.radar_timestamp_pairs = [
            (index, timestamp)
            for index, path in enumerate(self.radar_files)
            if (timestamp := _timestamp_from_stem_or_none(path.stem)) is not None
        ]
        self.timeline = [self._build_summary(index, path) for index, path in enumerate(self.lidar_files)]
        self.fps = self._estimate_fps()
        self.calibration = {
            "dataset_root": str(self.dataset_root),
            "ramap_root": str(self.dataset_root / "ramaps"),
            "radar_preview_source": "ramaps image files",
            "camera_rgb_intrinsics": {
                "width": 2304,
                "height": 1296,
                "K": RORAD_CAMERA_INTRINSICS.reshape(-1).tolist(),
                "P": RORAD_CAMERA_PROJECTION.reshape(-1).tolist(),
                "coordinate_frame": "image_camera_right_down_forward",
            },
            "camera_rgb_projection_matrix": RORAD_CAMERA_PROJECTION.tolist(),
            "camera_rgb_lidar_to_camera": RORAD_LIDAR_TO_CAMERA_PROJECTION.tolist(),
            "camera_rgb_lidar_to_camera_source_frame": "camera_flu_forward_left_up",
            "camera_rgb_lidar_origin_in_camera_flu": RORAD_LIDAR_ORIGIN_IN_CAMERA_FLU.tolist(),
            "camera_rgb_frame_id": "rgb",
            "lidar_frame_id": "lidar",
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
                index=self.camera_index_by_path.get(camera_path) if camera_path else None,
                timestamp=timestamp if camera_path else None,
                path=str(camera_path) if camera_path else None,
                delta_ms=0.0 if camera_path else None,
                available=camera_path is not None,
            )
        }

    def _build_radar_match(self, lidar_path: Path, timestamp: float) -> dict[str, SensorMatch]:
        radar_path = self.radar_by_stem.get(lidar_path.stem)
        radar_index = self.radar_index_by_path.get(radar_path) if radar_path else None
        radar_timestamp = timestamp if radar_path else None
        delta_ms = 0.0 if radar_path else None

        if radar_path is None and self.radar_timestamp_pairs:
            best_index, best_timestamp = min(
                self.radar_timestamp_pairs,
                key=lambda item: abs(float(item[1]) - float(timestamp)),
            )
            best_delta_ms = abs(float(best_timestamp) - float(timestamp)) * 1000.0
            if best_delta_ms <= self.config.radar_tolerance_ms:
                radar_index = best_index
                radar_path = self.radar_files[best_index]
                radar_timestamp = best_timestamp
                delta_ms = best_delta_ms

        return {
            "heatmap": SensorMatch(
                index=radar_index,
                timestamp=radar_timestamp,
                path=str(radar_path) if radar_path else None,
                delta_ms=delta_ms,
                available=radar_path is not None,
            )
        }

    def _build_summary(self, index: int, lidar_path: Path) -> dict:
        timestamp = _timestamp_from_stem(lidar_path.stem)
        return {
            "frame_index": index,
            "timestamp": timestamp,
            "lidar_path": str(lidar_path),
            "camera": self._build_camera_match(lidar_path, timestamp),
            "radar": self._build_radar_match(lidar_path, timestamp),
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


ADAPTER_CLASS = RoRadAdapter
