#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/datasets/coloradarplus/dataset.py
# @IDE     : vscode

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.adapters.base import DatasetAdapter, DatasetMetadata, FrameSample
from app.models import DatasetConfig, DatasetSceneOption, SensorMatch
from app.utils import nearest_match, parse_vector_quaternion_file, read_timestamps

COLORADARPLUS_DEFAULT_SCENE_ID = "ec_courtyard_run0"
COLORADARPLUS_CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
COLORADARPLUS_ROOT = Path("/workspace-data")

DATASET_DEFINITION = {
    "id": "coloradarplus",
    "name": "ColoradarPlus",
    "class_names": COLORADARPLUS_CLASS_NAMES,
    "default_scene_id": COLORADARPLUS_DEFAULT_SCENE_ID,
}


def _coloradar_scene_root(scene_id: str) -> Path:
    return COLORADARPLUS_ROOT / scene_id


def list_dataset_scenes() -> list[DatasetSceneOption]:
    scene_root = _coloradar_scene_root(COLORADARPLUS_DEFAULT_SCENE_ID)
    lidar_root = scene_root / "lidar" / "pointclouds"
    available = lidar_root.exists() and next(lidar_root.glob("*.bin"), None) is not None
    reason = None if available else f"{COLORADARPLUS_DEFAULT_SCENE_ID} is not mounted inside the container"
    return [
        DatasetSceneOption(
            id=COLORADARPLUS_DEFAULT_SCENE_ID,
            name=COLORADARPLUS_DEFAULT_SCENE_ID,
            available=available,
            reason=reason,
        )
    ]


def build_dataset_config(scene_id: str) -> DatasetConfig:
    return DatasetConfig(
        adapter="coloradarplus",
        dataset_root=str(_coloradar_scene_root(scene_id)),
        calib_root=str(COLORADARPLUS_ROOT / "calib"),
        sequence_name=scene_id,
        class_names=COLORADARPLUS_CLASS_NAMES,
        camera_tolerance_ms=75.0,
        radar_tolerance_ms=125.0,
    )


class ColoradarPlusAdapter(DatasetAdapter):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.calib_root = Path(config.calib_root or "")

        self.lidar_files = sorted((self.dataset_root / "lidar" / "pointclouds").glob("*.bin"))
        self.lidar_timestamps = read_timestamps(self.dataset_root / "lidar" / "timestamps.txt")
        self.rgb_files = sorted((self.dataset_root / "camera" / "images" / "rgb").glob("*.png"))
        self.rgb_timestamps = read_timestamps(self.dataset_root / "camera" / "rgb_timestamps.txt")
        self.radar_files = sorted((self.dataset_root / "cascade" / "heatmaps" / "data").glob("*.bin"))
        self.radar_timestamps = read_timestamps(self.dataset_root / "cascade" / "heatmaps" / "timestamps.txt")
        self.pose_timestamps = read_timestamps(self.dataset_root / "groundtruth" / "timestamps.txt")
        self.pose_rows = [
            [float(value) for value in row.split()]
            for row in (self.dataset_root / "groundtruth" / "groundtruth_poses.txt").read_text(encoding="utf-8").splitlines()
            if row.strip()
        ]

        self.calibration = self._load_calibration()
        self.timeline = [self._build_summary(idx) for idx in range(min(len(self.lidar_files), len(self.lidar_timestamps)))]
        self.fps = self._estimate_fps()

    def __len__(self) -> int:
        return len(self.timeline)

    def _estimate_fps(self) -> float:
        if len(self.lidar_timestamps) < 2:
            return 0.0
        deltas = np.diff(np.array(self.lidar_timestamps[: len(self.timeline)], dtype=np.float64))
        mean_delta = float(np.mean(deltas)) if deltas.size else 0.0
        return 1.0 / mean_delta if mean_delta > 0 else 0.0

    def _load_calibration(self) -> dict:
        transforms_root = self.calib_root / "transforms"
        calibration: dict[str, object] = {}
        if transforms_root.exists():
            for transform_path in transforms_root.glob("*.txt"):
                calibration[transform_path.stem] = parse_vector_quaternion_file(transform_path)
        rgb_info_path = self.calib_root / "camera" / "rgb_cam_info.json"
        if rgb_info_path.exists():
            intrinsics = json.loads(rgb_info_path.read_text(encoding="utf-8"))
            if isinstance(intrinsics, str):
                intrinsics = json.loads(intrinsics)
            calibration["camera_rgb_intrinsics"] = intrinsics
        return calibration

    def _ego_pose(self, lidar_timestamp: float) -> list[list[float]] | None:
        index, _ = nearest_match(lidar_timestamp, self.pose_timestamps, tolerance_ms=1000.0)
        if index is None or index >= len(self.pose_rows):
            return None
        tx, ty, tz, qx, qy, qz, qw = self.pose_rows[index][:7]
        qx2, qy2, qz2, qw2 = qx * qx, qy * qy, qz * qz, qw * qw
        rotation = np.array(
            [
                [qw2 + qx2 - qy2 - qz2, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), qw2 - qx2 + qy2 - qz2, 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw2 - qx2 - qy2 + qz2],
            ],
            dtype=np.float64,
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = [tx, ty, tz]
        return transform.tolist()

    def _build_camera_match(self, reference_ts: float) -> dict[str, SensorMatch]:
        index, delta_ms = nearest_match(reference_ts, self.rgb_timestamps, self.config.camera_tolerance_ms)
        path = str(self.rgb_files[index]) if index is not None and index < len(self.rgb_files) else None
        timestamp = self.rgb_timestamps[index] if index is not None else None
        return {
            "rgb": SensorMatch(
                index=index,
                timestamp=timestamp,
                path=path,
                delta_ms=delta_ms,
                available=index is not None and path is not None,
            )
        }

    def _build_radar_match(self, reference_ts: float) -> dict[str, SensorMatch]:
        index, delta_ms = nearest_match(reference_ts, self.radar_timestamps, self.config.radar_tolerance_ms)
        path = str(self.radar_files[index]) if index is not None and index < len(self.radar_files) else None
        timestamp = self.radar_timestamps[index] if index is not None else None
        return {
            "heatmap": SensorMatch(
                index=index,
                timestamp=timestamp,
                path=path,
                delta_ms=delta_ms,
                available=index is not None and path is not None,
            )
        }

    def _build_summary(self, index: int) -> dict:
        lidar_path = str(self.lidar_files[index])
        timestamp = self.lidar_timestamps[index]
        return {
            "frame_index": index,
            "timestamp": timestamp,
            "lidar_path": lidar_path,
            "camera": self._build_camera_match(timestamp),
            "radar": self._build_radar_match(timestamp),
        }

    def get_frame_summary(self, index: int) -> dict:
        return self.timeline[index]

    def get_ego_pose(self, index: int) -> list[list[float]] | None:
        summary = self.timeline[index]
        return self._ego_pose(summary["timestamp"])

    def get_frame(self, index: int) -> FrameSample:
        summary = self.timeline[index]
        lidar_points = np.fromfile(summary["lidar_path"], dtype=np.float32).reshape(-1, 4)
        return FrameSample(
            frame_index=index,
            timestamp=summary["timestamp"],
            lidar_points=lidar_points,
            lidar_path=summary["lidar_path"],
            camera_frames=summary["camera"],
            radar_frames=summary["radar"],
            calibration=self.calibration,
            ego_pose=self._ego_pose(summary["timestamp"]),
        )

    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            sequence_name=self.config.sequence_name,
            class_names=self.config.class_names,
            fps=self.fps,
            calibration=self.calibration,
        )


ADAPTER_CLASS = ColoradarPlusAdapter
