from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from app.models import Box3D, Vector3

TRACKING_CLASS_CONFIG = {
    "vehicle": {"max_distance_xy": 4.5, "max_z_delta": 2.0},
    "pedestrian": {"max_distance_xy": 1.6, "max_z_delta": 1.6},
    "bicycle": {"max_distance_xy": 2.2, "max_z_delta": 1.8},
}
DEFAULT_TRACKING_CONFIG = {"max_distance_xy": 3.0, "max_z_delta": 1.8}

TrackingMethod = Literal["greedy_cv"]
TrackingProgressCallback = Callable[[int, int, int], None]


class TrackingCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class TrackingConfig:
    method: TrackingMethod = "greedy_cv"
    input_score_threshold: float | None = None
    max_gap_frames: int = 2
    max_track_lost_frames: int = 2
    match_distance_xy: float | None = None
    match_z_delta: float | None = None


@dataclass
class FrameTrackingInput:
    frame_index: int
    timestamp: float
    annotations: list[Box3D]


@dataclass
class TrackObservation:
    frame_index: int
    timestamp: float
    annotation: Box3D


@dataclass
class TrackState:
    track_id: str
    class_name: str
    last_annotation: Box3D
    last_frame_index: int
    last_timestamp: float
    velocity_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    observations: list[TrackObservation] = field(default_factory=list)


class TrackingService:
    def track(
        self,
        frames: list[FrameTrackingInput],
        config: TrackingConfig,
        *,
        progress_callback: TrackingProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[int, list[Box3D]]:
        if config.method == "greedy_cv":
            return self._track_greedy_cv(
                frames,
                config,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
        raise ValueError(f"Unsupported tracking method: {config.method}")

    def _track_greedy_cv(
        self,
        frames: list[FrameTrackingInput],
        config: TrackingConfig,
        *,
        progress_callback: TrackingProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[int, list[Box3D]]:
        frame_results = {frame.frame_index: [] for frame in frames}
        active_tracks: dict[str, list[TrackState]] = {}
        all_tracks: list[TrackState] = []
        next_track_number = 1
        total_frames = len(frames)

        for completed_frames, frame in enumerate(frames, start=1):
            self._ensure_not_cancelled(cancel_event)
            for class_name, detections in self._group_detections(frame.annotations, config).items():
                class_tracks = active_tracks.setdefault(class_name, [])
                class_tracks[:] = [
                    track
                    for track in class_tracks
                    if frame.frame_index - track.last_frame_index <= config.max_track_lost_frames + 1
                ]
                matched_detection_indexes = self._associate_tracks(
                    class_tracks=class_tracks,
                    detections=detections,
                    frame_index=frame.frame_index,
                    timestamp=frame.timestamp,
                    config=config,
                )
                for detection_index, detection in enumerate(detections):
                    if detection_index in matched_detection_indexes:
                        continue
                    track = self._create_track(
                        track_number=next_track_number,
                        class_name=class_name,
                        frame_index=frame.frame_index,
                        timestamp=frame.timestamp,
                        detection=detection,
                    )
                    next_track_number += 1
                    class_tracks.append(track)
                    all_tracks.append(track)

            for class_name, class_tracks in active_tracks.items():
                active_tracks[class_name] = [
                    track
                    for track in class_tracks
                    if frame.frame_index - track.last_frame_index <= config.max_track_lost_frames + 1
                ]
            if progress_callback is not None:
                progress_callback(completed_frames, total_frames, frame.frame_index)

        self._ensure_not_cancelled(cancel_event)
        for track in all_tracks:
            observations = sorted(track.observations, key=lambda item: item.frame_index)
            for observation in observations:
                frame_results[observation.frame_index].append(
                    observation.annotation.model_copy(
                        update={
                            "track_id": track.track_id,
                            "source_stage": "manual"
                            if observation.annotation.source_model == "manual"
                            else "detected",
                        }
                    )
                )
            for previous, current in zip(observations, observations[1:]):
                gap = current.frame_index - previous.frame_index - 1
                if gap <= 0 or gap > config.max_gap_frames:
                    continue
                for offset in range(1, gap + 1):
                    missing_frame = previous.frame_index + offset
                    alpha = offset / (gap + 1)
                    frame_results[missing_frame].append(
                        self._interpolate_annotation(
                            track_id=track.track_id,
                            previous=previous.annotation,
                            current=current.annotation,
                            alpha=alpha,
                            frame_index=missing_frame,
                        )
                    )

        return frame_results

    def _group_detections(self, annotations: list[Box3D], config: TrackingConfig) -> dict[str, list[Box3D]]:
        grouped: dict[str, list[Box3D]] = {}
        for annotation in annotations:
            if annotation.source_model == "manual" or annotation.source_stage == "interpolated":
                continue
            if config.input_score_threshold is not None and annotation.score < config.input_score_threshold:
                continue
            grouped.setdefault(annotation.class_name, []).append(annotation)
        return grouped

    def _associate_tracks(
        self,
        *,
        class_tracks: list[TrackState],
        detections: list[Box3D],
        frame_index: int,
        timestamp: float,
        config: TrackingConfig,
    ) -> set[int]:
        candidate_pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(class_tracks):
            predicted_xy = self._predict_xy(track, timestamp)
            max_distance_xy = self._max_distance_xy(track.class_name, frame_index - track.last_frame_index, config)
            max_z_delta = self._max_z_delta(track.class_name, config)
            for detection_index, detection in enumerate(detections):
                detection_xy = np.asarray([detection.center.x, detection.center.y], dtype=np.float32)
                distance_xy = float(np.linalg.norm(predicted_xy - detection_xy))
                z_delta = abs(track.last_annotation.center.z - detection.center.z)
                if distance_xy > max_distance_xy or z_delta > max_z_delta:
                    continue
                size_delta = self._size_delta(track.last_annotation, detection)
                cost = distance_xy + 0.35 * z_delta + 0.5 * size_delta
                candidate_pairs.append((cost, track_index, detection_index))

        matched_track_indexes: set[int] = set()
        matched_detection_indexes: set[int] = set()
        for _, track_index, detection_index in sorted(candidate_pairs, key=lambda item: item[0]):
            if track_index in matched_track_indexes or detection_index in matched_detection_indexes:
                continue
            track = class_tracks[track_index]
            detection = detections[detection_index]
            self._update_track(
                track=track,
                frame_index=frame_index,
                timestamp=timestamp,
                detection=detection,
            )
            matched_track_indexes.add(track_index)
            matched_detection_indexes.add(detection_index)
        return matched_detection_indexes

    def _create_track(
        self,
        *,
        track_number: int,
        class_name: str,
        frame_index: int,
        timestamp: float,
        detection: Box3D,
    ) -> TrackState:
        track_id = f"trk-{track_number:05d}"
        return TrackState(
            track_id=track_id,
            class_name=class_name,
            last_annotation=detection,
            last_frame_index=frame_index,
            last_timestamp=timestamp,
            observations=[TrackObservation(frame_index=frame_index, timestamp=timestamp, annotation=detection)],
        )

    def _update_track(
        self,
        *,
        track: TrackState,
        frame_index: int,
        timestamp: float,
        detection: Box3D,
    ) -> None:
        previous_xy = np.asarray([track.last_annotation.center.x, track.last_annotation.center.y], dtype=np.float32)
        current_xy = np.asarray([detection.center.x, detection.center.y], dtype=np.float32)
        delta_t = max(timestamp - track.last_timestamp, 1e-3)
        track.velocity_xy = (current_xy - previous_xy) / delta_t
        track.last_annotation = detection
        track.last_frame_index = frame_index
        track.last_timestamp = timestamp
        track.observations.append(
            TrackObservation(
                frame_index=frame_index,
                timestamp=timestamp,
                annotation=detection,
            )
        )

    def _predict_xy(self, track: TrackState, timestamp: float) -> np.ndarray:
        delta_t = max(timestamp - track.last_timestamp, 0.0)
        current_xy = np.asarray([track.last_annotation.center.x, track.last_annotation.center.y], dtype=np.float32)
        if delta_t <= 0:
            return current_xy
        return current_xy + track.velocity_xy * delta_t

    def _max_distance_xy(self, class_name: str, frame_gap: int, config: TrackingConfig) -> float:
        if config.match_distance_xy is not None:
            return float(config.match_distance_xy) * max(1, frame_gap)
        class_config = TRACKING_CLASS_CONFIG.get(class_name.lower(), DEFAULT_TRACKING_CONFIG)
        return float(class_config["max_distance_xy"]) * max(1, frame_gap)

    def _max_z_delta(self, class_name: str, config: TrackingConfig) -> float:
        if config.match_z_delta is not None:
            return float(config.match_z_delta)
        class_config = TRACKING_CLASS_CONFIG.get(class_name.lower(), DEFAULT_TRACKING_CONFIG)
        return float(class_config["max_z_delta"])

    def _size_delta(self, previous: Box3D, current: Box3D) -> float:
        previous_size = np.asarray([previous.size.x, previous.size.y, previous.size.z], dtype=np.float32)
        current_size = np.asarray([current.size.x, current.size.y, current.size.z], dtype=np.float32)
        baseline = np.maximum(previous_size, 1e-3)
        return float(np.mean(np.abs(current_size - previous_size) / baseline))

    def _interpolate_annotation(
        self,
        *,
        track_id: str,
        previous: Box3D,
        current: Box3D,
        alpha: float,
        frame_index: int,
    ) -> Box3D:
        previous_center = np.asarray([previous.center.x, previous.center.y, previous.center.z], dtype=np.float32)
        current_center = np.asarray([current.center.x, current.center.y, current.center.z], dtype=np.float32)
        previous_size = np.asarray([previous.size.x, previous.size.y, previous.size.z], dtype=np.float32)
        current_size = np.asarray([current.size.x, current.size.y, current.size.z], dtype=np.float32)
        center = previous_center + (current_center - previous_center) * alpha
        size = previous_size + (current_size - previous_size) * alpha
        yaw = self._interpolate_angle(previous.yaw, current.yaw, alpha)

        return Box3D(
            id=f"{track_id}-interp-{frame_index:06d}",
            class_name=previous.class_name,
            score=min(previous.score, current.score),
            center=Vector3(x=float(center[0]), y=float(center[1]), z=float(center[2])),
            size=Vector3(x=float(size[0]), y=float(size[1]), z=float(size[2])),
            yaw=yaw,
            source_model=previous.source_model,
            coord_frame="lidar",
            track_id=track_id,
            source_stage="interpolated",
        )

    def _interpolate_angle(self, previous: float, current: float, alpha: float) -> float:
        delta = math.atan2(math.sin(current - previous), math.cos(current - previous))
        return previous + delta * alpha

    def _ensure_not_cancelled(self, cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise TrackingCancelled("Tracking cancelled")
