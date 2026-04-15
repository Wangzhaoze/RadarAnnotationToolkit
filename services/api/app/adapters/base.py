#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/api/app/adapters/base.py
# @IDE     : vscode

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.models import SensorMatch


def sensor_match_get(match: SensorMatch | dict | None, key: str, default=None):
    if match is None:
        return default
    if isinstance(match, dict):
        return match.get(key, default)
    return getattr(match, key, default)


@dataclass
class FrameSample:
    frame_index: int
    timestamp: float
    lidar_points: np.ndarray
    lidar_path: str
    camera_frames: dict[str, SensorMatch]
    radar_frames: dict[str, SensorMatch]
    calibration: dict
    ego_pose: list[list[float]] | None
    lidar_semantic_labels: np.ndarray | None = None


@dataclass
class DatasetMetadata:
    sequence_name: str
    class_names: list[str]
    fps: float
    calibration: dict


class DatasetAdapter(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_frame(self, index: int) -> FrameSample:
        raise NotImplementedError

    @abstractmethod
    def get_frame_summary(self, index: int) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> DatasetMetadata:
        raise NotImplementedError

    def on_loaded(self) -> None:
        return None

    def get_ego_pose(self, index: int) -> list[list[float]] | None:
        return self.get_frame(index).ego_pose

    def get_semantic_summary(self, index: int) -> dict:
        return {"lidar": False, "camera": {}, "radar": {}}

    def get_lidar_semantic_labels(self, index: int) -> np.ndarray | None:
        return self.get_frame(index).lidar_semantic_labels

    def get_camera_semantic_path(self, index: int, camera_name: str) -> Path | str | None:
        return None

    def get_radar_semantic_path(self, index: int, radar_name: str) -> Path | str | None:
        return None

    def get_radar_preview_path(self, index: int, radar_name: str) -> Path | str | None:
        summary = self.get_frame_summary(index)
        match = summary.get("radar", {}).get(radar_name)
        if not match or not sensor_match_get(match, "available") or not sensor_match_get(match, "path"):
            return None
        return sensor_match_get(match, "path")

    def get_ramap_cache_status(self) -> dict | None:
        return None

    def get_layer_defaults(self, index: int) -> dict:
        return {"lidar": None, "camera": {}, "radar": {}}
