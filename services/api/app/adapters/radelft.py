from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

from app.adapters.base import DatasetAdapter, DatasetMetadata, FrameSample, sensor_match_get
from app.models import DatasetConfig, DatasetSceneOption
from app.services.ramap_cache import RAMAP_DIR_NAME, is_ready_ramap, ramap_cache_manager, ramap_path_for_cube
from app.utils import nearest_match, timestamp_from_path

RADELFT_CLASS_NAMES = ["Vehicle", "Pedestrian", "Bicycle"]
RADELFT_DEFAULT_SCENE_ID = "Scene6"
RADELFT_SCENE_IDS = [f"Scene{index}" for index in range(1, 7)]
RADELFT_ROOT = Path("/workspace-data/radelft")
RADELFT_POSE_TOLERANCE_MS = 200.0

DATASET_DEFINITION = {
    "id": "radelft",
    "name": "RaDelft",
    "class_names": RADELFT_CLASS_NAMES,
    "default_scene_id": RADELFT_DEFAULT_SCENE_ID,
}


def _radelft_scene_root(scene_id: str) -> Path:
    return RADELFT_ROOT / scene_id


def _radar_cube_frame_number(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except (TypeError, ValueError):
        return 0


def _load_radar_cube_timestamps(radar_cube_root: Path, radar_files: list[Path]) -> list[float]:
    timestamps_path = radar_cube_root / "timestamps.mat"
    if not timestamps_path.exists() or not radar_files:
        return []

    import scipy.io

    payload = scipy.io.loadmat(timestamps_path, variable_names=["unixDateTime"])
    values = np.asarray(payload.get("unixDateTime"), dtype=np.float64).reshape(-1)
    timestamps: list[float] = []
    for path in radar_files:
        frame_number = _radar_cube_frame_number(path)
        timestamp_index = frame_number - 1
        if timestamp_index < 0 or timestamp_index >= values.size:
            continue
        timestamps.append(float(values[timestamp_index]))
    return timestamps


def _nearest_match_from_sorted_array(
    reference_ts: float,
    candidates: np.ndarray,
    tolerance_ms: float,
) -> tuple[int | None, float | None]:
    if candidates.size == 0:
        return None, None
    reference = float(reference_ts)
    insertion_index = int(np.searchsorted(candidates, reference))
    candidate_indices = []
    if insertion_index > 0:
        candidate_indices.append(insertion_index - 1)
    if insertion_index < candidates.size:
        candidate_indices.append(insertion_index)
    if not candidate_indices:
        return None, None
    best_index = min(candidate_indices, key=lambda candidate_index: abs(float(candidates[candidate_index]) - reference))
    delta_ms = abs(float(candidates[best_index]) - reference) * 1000.0
    if delta_ms > tolerance_ms:
        return None, delta_ms
    return best_index, delta_ms


def _probe_radelft_scene(scene_id: str) -> tuple[bool, str | None]:
    scene_root = _radelft_scene_root(scene_id)
    if not scene_root.exists():
        return False, f"{scene_id} is not mounted inside the container"
    ros_root = scene_root / "rosDS"
    lidar_root = ros_root / "rslidar_points"
    if not lidar_root.exists():
        return False, f"{scene_id} is missing rosDS/rslidar_points"
    first_lidar = next(lidar_root.glob("*.npy"), None)
    if first_lidar is None:
        return False, f"{scene_id} does not contain LiDAR frames"
    try:
        sample = np.load(first_lidar, mmap_mode="r")
        if getattr(sample, "ndim", 0) < 1:
            return False, f"{scene_id} LiDAR frames are malformed"
    except Exception as exc:
        return False, f"{scene_id} failed LiDAR preload: {exc}"
    return True, None


def list_dataset_scenes() -> list[DatasetSceneOption]:
    scenes: list[DatasetSceneOption] = []
    for scene_id in RADELFT_SCENE_IDS:
        available, reason = _probe_radelft_scene(scene_id)
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
        adapter="radelft",
        dataset_root=str(_radelft_scene_root(scene_id)),
        sequence_name=scene_id,
        class_names=RADELFT_CLASS_NAMES,
        camera_tolerance_ms=50.0,
        radar_tolerance_ms=125.0,
    )


def _build_radelft_camera_transform() -> np.ndarray:
    # Match the hand-tuned transform used by the official camera projection visualizer.
    roll = np.radians(0.0)
    pitch = np.radians(-85.0)
    yaw = np.radians(90.0)

    rotation_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float32,
    )
    rotation_y = np.asarray(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ],
        dtype=np.float32,
    )
    rotation_z = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    rotation = rotation_z @ rotation_y @ rotation_x
    extra_pitch = np.radians(1.0)
    rotation_h = np.asarray(
        [
            [np.cos(extra_pitch), 0.0, np.sin(extra_pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(extra_pitch), 0.0, np.cos(extra_pitch)],
        ],
        dtype=np.float32,
    )
    rotation = rotation_h @ rotation

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray([0.195, 0.207, -0.482], dtype=np.float32)
    return transform


RADELFT_LIDAR_TO_CAMERA_RGB = _build_radelft_camera_transform()

RADELFT_CAMERA_RGB_PROJECTION = np.asarray(
    [
        [1422.994677, 0.0, 943.853027, 0.0],
        [0.0, 1422.994677, 608.537918, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def _quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _pose_message_to_transform(message: dict) -> np.ndarray | None:
    try:
        pose_container = message.get("pose") if isinstance(message, dict) else None
        pose_payload = (pose_container or {}).get("pose") if isinstance(pose_container, dict) else None
        if not isinstance(pose_payload, dict):
            return None
        position_payload = pose_payload.get("position") or {}
        orientation_payload = pose_payload.get("orientation") or {}
        if not isinstance(position_payload, dict) or not isinstance(orientation_payload, dict):
            return None
        translation = np.asarray(
            [
                float(position_payload.get("x", 0.0)),
                float(position_payload.get("y", 0.0)),
                float(position_payload.get("z", 0.0)),
            ],
            dtype=np.float64,
        )
        rotation = _quaternion_to_rotation_matrix(
            float(orientation_payload.get("x", 0.0)),
            float(orientation_payload.get("y", 0.0)),
            float(orientation_payload.get("z", 0.0)),
            float(orientation_payload.get("w", 1.0)),
        )
    except (AttributeError, TypeError, ValueError):
        return None
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _load_lidar_points(path: Path) -> np.ndarray:
    points = np.load(path)
    if getattr(points.dtype, "names", None):
        points = structured_to_unstructured(points)

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"Expected LiDAR array with shape (N, 3+) in {path}")

    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]

    if points.shape[1] >= 4:
        points = points[:, :4]
    elif points.shape[1] == 3:
        intensity = np.linalg.norm(points, axis=1, keepdims=True).astype(np.float32)
        points = np.concatenate([points, intensity], axis=1)
    else:
        raise ValueError(f"Expected LiDAR array with shape (N, 3+) in {path}")

    if points.shape[0]:
        intensity = points[:, 3]
        intensity_min = float(intensity.min())
        intensity_max = float(intensity.max())
        if intensity_max > intensity_min:
            points[:, 3] = (intensity - intensity_min) / (intensity_max - intensity_min + 1e-6)

    max_points = 120000
    if points.shape[0] > max_points:
        sample_indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[sample_indices]

    return points.astype(np.float32, copy=False)


class RaDelftAdapter(DatasetAdapter):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.ros_root = self.dataset_root / "rosDS"
        self.pose_timestamps: list[float] = []
        self.pose_matrices: list[list[list[float]]] = []
        self.pose_frame_id: str | None = None
        self.pose_child_frame_id: str | None = None
        self.pose_path = self.ros_root / "gps_odom.topic.json"
        self._pose_loaded = False

        self.lidar_files = sorted((self.ros_root / "rslidar_points").glob("*.npy"))
        self.camera_files = sorted((self.ros_root / "ueye_left_image_rect_color").glob("*.jpg"))

        radar_cube_root = self.dataset_root / "RadarCubes"
        if not radar_cube_root.exists():
            radar_cube_root = self.dataset_root / "radarcubes"
        self.radar_files = sorted(radar_cube_root.glob("Pow_Frame_*.mat"), key=_radar_cube_frame_number)
        self.radar_semantic_files = sorted((self.dataset_root / "radar_gt").glob("*.npy"))
        self.ramap_root = self.dataset_root / RAMAP_DIR_NAME

        if not self.lidar_files:
            raise ValueError(f"No LiDAR frames found under {self.ros_root / 'rslidar_points'}")

        self.lidar_timestamps = [timestamp_from_path(path) for path in self.lidar_files]
        self.camera_timestamps = [timestamp_from_path(path) for path in self.camera_files]
        self.radar_timestamps = _load_radar_cube_timestamps(radar_cube_root, self.radar_files)
        if len(self.radar_timestamps) != len(self.radar_files):
            self.radar_files = []
            self.radar_timestamps = []
        self.radar_semantic_timestamps = [timestamp_from_path(path) for path in self.radar_semantic_files]
        self.camera_timestamps_array = np.asarray(self.camera_timestamps, dtype=np.float64)
        self.radar_timestamps_array = np.asarray(self.radar_timestamps, dtype=np.float64)
        self.radar_semantic_timestamps_array = np.asarray(self.radar_semantic_timestamps, dtype=np.float64)
        self.calibration = self._load_calibration()
        self.timeline = []
        for frame_index, lidar_path in enumerate(self.lidar_files):
            self.timeline.append(self._build_summary(frame_index, lidar_path, self.lidar_timestamps[frame_index]))
        self.fps = self._estimate_fps()

    def __len__(self) -> int:
        return len(self.timeline)

    def on_loaded(self) -> None:
        if self.radar_files:
            ramap_cache_manager.ensure_cache(
                sequence_name=self.config.sequence_name,
                radar_files=self.radar_files,
                ramap_root=self.ramap_root,
            )

    def _estimate_fps(self) -> float:
        if len(self.lidar_timestamps) < 2:
            return 0.0
        deltas = np.diff(np.asarray(self.lidar_timestamps, dtype=np.float64))
        mean_delta = float(np.mean(deltas)) if deltas.size else 0.0
        return 1.0 / mean_delta if mean_delta > 0 else 0.0

    def _load_calibration(self) -> dict:
        calibration: dict[str, object] = {}
        camera_info_path = self.ros_root / "ueye_left_camera_info.topic.json"
        if camera_info_path.exists():
            payload = json.loads(camera_info_path.read_text(encoding="utf-8"))
            messages = payload.get("messages") or []
            if messages:
                first_message = messages[0]
                calibration["camera_rgb_intrinsics"] = {
                    "width": first_message.get("width"),
                    "height": first_message.get("height"),
                    "distortion_model": first_message.get("distortion_model"),
                    "D": first_message.get("D"),
                    "K": first_message.get("K"),
                    "R": first_message.get("R"),
                    "P": first_message.get("P"),
                }
        calibration["camera_rgb_projection_matrix"] = RADELFT_CAMERA_RGB_PROJECTION.tolist()
        calibration["camera_rgb_lidar_to_camera"] = RADELFT_LIDAR_TO_CAMERA_RGB.tolist()
        calibration["camera_rgb_frame_id"] = "left"
        calibration["lidar_frame_id"] = "rslidar"
        if self.pose_path.exists():
            calibration["world_frame_id"] = self.pose_frame_id or "utm"
            calibration["ego_pose_local_frame_id"] = self.pose_child_frame_id or "an_device"
            calibration["ego_pose_source"] = "gps_odom.topic.json"
        calibration["dataset_root"] = str(self.dataset_root)
        return calibration

    def _load_pose_stream(self) -> None:
        if self._pose_loaded:
            return
        self._pose_loaded = True
        if not self.pose_path.exists():
            return
        payload = json.loads(self.pose_path.read_text(encoding="utf-8"))
        messages = payload.get("messages") or []
        pose_timestamps: list[float] = []
        pose_matrices: list[list[list[float]]] = []
        for message in messages:
            timestamp_ns = message.get("_timestamp_ns")
            if timestamp_ns is None:
                continue
            transform = _pose_message_to_transform(message)
            if transform is None:
                continue
            pose_timestamps.append(float(timestamp_ns) / 1_000_000_000.0)
            pose_matrices.append(transform.tolist())
            if self.pose_frame_id is None:
                self.pose_frame_id = (message.get("header") or {}).get("frame_id")
            if self.pose_child_frame_id is None:
                self.pose_child_frame_id = message.get("child_frame_id")
        self.pose_timestamps = pose_timestamps
        self.pose_matrices = pose_matrices

    def _ego_pose(self, lidar_timestamp: float) -> list[list[float]] | None:
        try:
            self._load_pose_stream()
        except Exception:
            return None
        index, _ = nearest_match(lidar_timestamp, self.pose_timestamps, RADELFT_POSE_TOLERANCE_MS)
        if index is None or index >= len(self.pose_matrices):
            return None
        return self.pose_matrices[index]

    def _build_camera_match(self, reference_ts: float) -> dict[str, dict]:
        index, delta_ms = _nearest_match_from_sorted_array(
            reference_ts,
            self.camera_timestamps_array,
            self.config.camera_tolerance_ms,
        )
        path = str(self.camera_files[index]) if index is not None and index < len(self.camera_files) else None
        timestamp = self.camera_timestamps[index] if index is not None else None
        return {
            "rgb": {
                "index": index,
                "timestamp": timestamp,
                "path": path,
                "delta_ms": delta_ms,
                "available": index is not None and path is not None,
            }
        }

    def _build_radar_match(self, reference_ts: float) -> dict[str, dict]:
        index, delta_ms = _nearest_match_from_sorted_array(
            reference_ts,
            self.radar_timestamps_array,
            self.config.radar_tolerance_ms,
        )
        path = str(self.radar_files[index]) if index is not None and index < len(self.radar_files) else None
        timestamp = self.radar_timestamps[index] if index is not None else None
        return {
            "heatmap": {
                "index": index,
                "timestamp": timestamp,
                "path": path,
                "delta_ms": delta_ms,
                "available": index is not None and path is not None,
            }
        }

    def _build_radar_semantic_match(self, reference_ts: float) -> dict[str, dict]:
        index, delta_ms = _nearest_match_from_sorted_array(
            reference_ts,
            self.radar_semantic_timestamps_array,
            self.config.radar_tolerance_ms,
        )
        path = str(self.radar_semantic_files[index]) if index is not None and index < len(self.radar_semantic_files) else None
        timestamp = self.radar_semantic_timestamps[index] if index is not None else None
        return {
            "heatmap": {
                "index": index,
                "timestamp": timestamp,
                "path": path,
                "delta_ms": delta_ms,
                "available": index is not None and path is not None,
            }
        }

    def _build_summary(self, index: int, lidar_path: Path, timestamp: float) -> dict:
        return {
            "frame_index": index,
            "timestamp": timestamp,
            "lidar_path": str(lidar_path),
            "camera": self._build_camera_match(timestamp),
            "radar": self._build_radar_match(timestamp),
            "radar_semantic": self._build_radar_semantic_match(timestamp),
        }

    def get_frame_summary(self, index: int) -> dict:
        return self.timeline[index]

    def get_semantic_summary(self, index: int) -> dict:
        summary = self.timeline[index]
        return {
            "lidar": False,
            "camera": {name: False for name in summary["camera"]},
            "radar": {
                name: bool(sensor_match_get(match, "available") and sensor_match_get(match, "path"))
                for name, match in summary.get("radar_semantic", {}).items()
            },
        }

    def get_radar_semantic_path(self, index: int, radar_name: str) -> Path | str | None:
        summary = self.timeline[index]
        match = summary.get("radar_semantic", {}).get(radar_name)
        if not match or not sensor_match_get(match, "available") or not sensor_match_get(match, "path"):
            return None
        return sensor_match_get(match, "path")

    def get_radar_preview_path(self, index: int, radar_name: str) -> Path | str | None:
        cache_status = self.get_ramap_cache_status()
        if not cache_status or not cache_status.get("intensity_available"):
            return None
        summary = self.timeline[index]
        match = summary.get("radar", {}).get(radar_name)
        match_index = sensor_match_get(match, "index")
        if not match or not sensor_match_get(match, "available") or match_index is None or match_index >= len(self.radar_files):
            return None
        ramap_path = ramap_path_for_cube(self.radar_files[match_index], self.ramap_root)
        return ramap_path if is_ready_ramap(ramap_path) else None

    def get_ramap_cache_status(self) -> dict | None:
        status = ramap_cache_manager.get_status(self.ramap_root)
        if status is not None:
            return status
        if not self.radar_files:
            return None
        total = len(self.radar_files)
        ready = sum(1 for path in self.radar_files if is_ready_ramap(ramap_path_for_cube(path, self.ramap_root)))
        return {
            "sequence_name": self.config.sequence_name,
            "ramap_root": str(self.ramap_root),
            "total_frames": total,
            "ready_frames": ready,
            "missing_frames": max(0, total - ready),
            "processed_frames": 0,
            "failed_frames": 0,
            "running": False,
            "workers": 0,
            "progress": ready / total if total else 1.0,
            "intensity_available": total > 0 and ready >= total,
            "started_at": None,
            "updated_at": None,
            "completed_at": None,
            "error": None,
            "failures": [],
        }

    def get_layer_defaults(self, index: int) -> dict:
        return {
            "lidar": None,
            "camera": {},
            "radar": {
                "heatmap": "semantic",
            },
        }

    def get_ego_pose(self, index: int) -> list[list[float]] | None:
        summary = self.timeline[index]
        return self._ego_pose(summary["timestamp"])

    def get_frame(self, index: int) -> FrameSample:
        summary = self.timeline[index]
        lidar_points = _load_lidar_points(Path(summary["lidar_path"]))
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
