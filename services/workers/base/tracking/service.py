#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/workers/base/tracking/service.py
# @IDE     : vscode

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Callable, Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.optimize import linear_sum_assignment


TrackingMethod = Literal["kalman_filter"]
TrackingProgressCallback = Callable[[int, int, int], None]
TrackingSourceStage = Literal["manual", "detected", "interpolated", "predicted"]


class TrackingCancelled(RuntimeError):
    pass


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class TrackingBox3D(BaseModel):
    id: str
    class_name: str
    score: float = 0.0
    center: Vector3
    size: Vector3
    yaw: float = 0.0
    source_model: str = "manual"
    coord_frame: Literal["lidar"] = "lidar"
    track_id: str | None = None
    source_stage: TrackingSourceStage | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_tracking_defaults(cls, value):
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if payload.get("source_stage") is None:
            payload["source_stage"] = "manual" if payload.get("source_model") == "manual" else "detected"
        if "track_id" not in payload:
            payload["track_id"] = None
        return payload


class TrackingConfig(BaseModel):
    method: TrackingMethod = "kalman_filter"
    input_score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    max_missed_frames: int = Field(default=3, ge=0)
    association_distance_m: float = Field(default=4.0, gt=0.0)
    process_noise_pos: float = Field(default=0.5, gt=0.0)
    process_noise_size: float = Field(default=0.1, gt=0.0)
    process_noise_vel: float = Field(default=1.0, gt=0.0)
    measurement_noise_pos: float = Field(default=1.0, gt=0.0)
    measurement_noise_size: float = Field(default=0.5, gt=0.0)
    yaw_smoothing_alpha: float = Field(default=0.5, ge=0.0, le=1.0)


class TrackingFrameInput(BaseModel):
    frame_index: int
    timestamp: float
    annotations: list[TrackingBox3D]
    ego_pose: list[list[float]] | None = None


class TrackingJobCreateRequest(BaseModel):
    config: TrackingConfig = Field(default_factory=TrackingConfig)
    frames: list[TrackingFrameInput]


class TrackingFrameResult(BaseModel):
    frame_index: int
    annotations: list[TrackingBox3D]


class TrackingJobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    total_frames: int
    completed_frames: int = 0
    current_frame: int | None = None
    error: str | None = None


class TrackingJobResultResponse(BaseModel):
    job_id: str
    method: TrackingMethod
    results: list[TrackingFrameResult]


@dataclass
class KalmanBoxFilter:
    x: np.ndarray
    p: np.ndarray

    @classmethod
    def from_observation(cls, observation: np.ndarray) -> KalmanBoxFilter:
        x = np.zeros(9, dtype=np.float64)
        x[:6] = observation
        p = np.eye(9, dtype=np.float64)
        p[:6, :6] *= 1.0
        p[6:, 6:] *= 10.0
        return cls(x=x, p=p)

    def predict(self, dt: float, config: TrackingConfig) -> None:
        dt = max(float(dt), 0.0)
        transition = np.eye(9, dtype=np.float64)
        transition[0, 6] = dt
        transition[1, 7] = dt
        transition[2, 8] = dt

        scale = max(dt, 1e-3)
        process_noise = np.diag(
            [
                config.process_noise_pos * scale,
                config.process_noise_pos * scale,
                config.process_noise_pos * scale,
                config.process_noise_size * scale,
                config.process_noise_size * scale,
                config.process_noise_size * scale,
                config.process_noise_vel * scale,
                config.process_noise_vel * scale,
                config.process_noise_vel * scale,
            ]
        ) ** 2

        self.x = transition @ self.x
        self.p = transition @ self.p @ transition.T + process_noise
        self.x[3:6] = np.maximum(self.x[3:6], 0.01)

    def update(self, observation: np.ndarray, config: TrackingConfig) -> None:
        observation_matrix = np.zeros((6, 9), dtype=np.float64)
        observation_matrix[:6, :6] = np.eye(6, dtype=np.float64)
        measurement_noise = np.diag(
            [
                config.measurement_noise_pos,
                config.measurement_noise_pos,
                config.measurement_noise_pos,
                config.measurement_noise_size,
                config.measurement_noise_size,
                config.measurement_noise_size,
            ]
        ) ** 2

        innovation = observation - observation_matrix @ self.x
        residual_covariance = observation_matrix @ self.p @ observation_matrix.T + measurement_noise
        kalman_gain = self.p @ observation_matrix.T @ np.linalg.inv(residual_covariance)
        self.x = self.x + kalman_gain @ innovation
        identity = np.eye(9, dtype=np.float64)
        self.p = (identity - kalman_gain @ observation_matrix) @ self.p
        self.x[3:6] = np.maximum(self.x[3:6], 0.01)

    def box_state(self) -> np.ndarray:
        return self.x[:6].copy()


@dataclass
class TrackingMeasurement:
    annotation: TrackingBox3D
    center: np.ndarray
    size: np.ndarray
    yaw: float

    @property
    def observation(self) -> np.ndarray:
        return np.concatenate([self.center, self.size]).astype(np.float64, copy=False)


@dataclass
class TrackState:
    track_id: str
    class_name: str
    filter: KalmanBoxFilter
    yaw: float
    last_timestamp: float
    state_timestamp: float
    last_annotation: TrackingBox3D
    missed_frames: int = 0


@dataclass
class TrackingCoordinateContext:
    use_world_frame: bool
    poses_by_frame: dict[int, np.ndarray]

    def pose_for_frame(self, frame_index: int) -> np.ndarray | None:
        if not self.use_world_frame:
            return None
        return self.poses_by_frame.get(frame_index)


@dataclass
class WorkerTrackingJob:
    job_id: str
    request: TrackingJobCreateRequest
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    completed_frames: int = 0
    current_frame: int | None = None
    error: str | None = None
    result: TrackingJobResultResponse | None = None

    @property
    def total_frames(self) -> int:
        return len(self.request.frames)

    def snapshot(self) -> TrackingJobStatusResponse:
        with self.lock:
            return TrackingJobStatusResponse(
                job_id=self.job_id,
                status=self.status,
                total_frames=self.total_frames,
                completed_frames=self.completed_frames,
                current_frame=self.current_frame,
                error=self.error,
            )

    def update_progress(self, completed_frames: int, total_frames: int, current_frame: int) -> None:
        with self.lock:
            self.completed_frames = completed_frames
            self.current_frame = current_frame

    def mark_running(self) -> None:
        with self.lock:
            self.status = "running"

    def mark_completed(self, result: TrackingJobResultResponse) -> None:
        with self.lock:
            self.status = "completed"
            self.completed_frames = self.total_frames
            self.current_frame = None
            self.result = result

    def mark_failed(self, exc: Exception) -> None:
        with self.lock:
            self.status = "failed"
            self.current_frame = None
            self.error = str(exc).strip() or exc.__class__.__name__

    def mark_cancelled(self) -> None:
        with self.lock:
            self.status = "cancelled"
            self.current_frame = None
            self.error = None


class TrackingService:
    def track(
        self,
        frames: list[TrackingFrameInput],
        config: TrackingConfig,
        *,
        progress_callback: TrackingProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[int, list[TrackingBox3D]]:
        if config.method != "kalman_filter":
            raise ValueError(f"Unsupported tracking method: {config.method}")
        return self._track_kalman(
            frames,
            config,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    def _track_kalman(
        self,
        frames: list[TrackingFrameInput],
        config: TrackingConfig,
        *,
        progress_callback: TrackingProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[int, list[TrackingBox3D]]:
        sorted_frames = sorted(frames, key=lambda item: (item.frame_index, item.timestamp))
        frame_results = {frame.frame_index: [] for frame in sorted_frames}
        coordinate_context = self._build_coordinate_context(sorted_frames)
        active_tracks: dict[str, list[TrackState]] = {}
        next_track_number = 1
        total_frames = len(sorted_frames)

        for completed_frames, frame in enumerate(sorted_frames, start=1):
            self._ensure_not_cancelled(cancel_event)
            frame_pose = coordinate_context.pose_for_frame(frame.frame_index)

            for class_tracks in active_tracks.values():
                for track in class_tracks:
                    self._predict_track_to_frame(track, frame.timestamp, config)

            grouped = self._group_measurements(frame, config, coordinate_context)
            matched_track_ids: set[str] = set()
            matched_detection_ids: set[str] = set()

            for class_name, measurements in grouped.items():
                class_tracks = active_tracks.setdefault(class_name, [])
                class_tracks[:] = [
                    track
                    for track in class_tracks
                    if track.missed_frames <= config.max_missed_frames
                ]
                matches = self._associate_tracks(class_tracks, measurements, config)
                for track_index, detection_index in matches:
                    track = class_tracks[track_index]
                    measurement = measurements[detection_index]
                    self._update_track(track, measurement, frame.timestamp, config)
                    frame_results[frame.frame_index].append(
                        self._annotation_from_track(
                            track=track,
                            frame_index=frame.frame_index,
                            pose=frame_pose,
                            source_annotation=measurement.annotation,
                            source_stage="detected",
                        )
                    )
                    matched_track_ids.add(track.track_id)
                    matched_detection_ids.add(measurement.annotation.id)

                for measurement in measurements:
                    if measurement.annotation.id in matched_detection_ids:
                        continue
                    track = self._create_track(
                        track_number=next_track_number,
                        measurement=measurement,
                        timestamp=frame.timestamp,
                    )
                    next_track_number += 1
                    class_tracks.append(track)
                    frame_results[frame.frame_index].append(
                        self._annotation_from_track(
                            track=track,
                            frame_index=frame.frame_index,
                            pose=frame_pose,
                            source_annotation=measurement.annotation,
                            source_stage="detected",
                        )
                    )
                    matched_track_ids.add(track.track_id)

            for class_name, class_tracks in list(active_tracks.items()):
                next_tracks: list[TrackState] = []
                for track in class_tracks:
                    if track.track_id in matched_track_ids:
                        next_tracks.append(track)
                        continue
                    track.missed_frames += 1
                    if track.missed_frames <= config.max_missed_frames:
                        frame_results[frame.frame_index].append(
                            self._annotation_from_track(
                                track=track,
                                frame_index=frame.frame_index,
                                pose=frame_pose,
                                source_annotation=track.last_annotation,
                                source_stage="predicted",
                            )
                        )
                        next_tracks.append(track)
                active_tracks[class_name] = next_tracks

            if progress_callback is not None:
                progress_callback(completed_frames, total_frames, frame.frame_index)

        return frame_results

    def _build_coordinate_context(self, frames: list[TrackingFrameInput]) -> TrackingCoordinateContext:
        poses_by_frame: dict[int, np.ndarray] = {}
        for frame in frames:
            pose = self._coerce_pose_matrix(frame.ego_pose)
            if pose is None:
                return TrackingCoordinateContext(use_world_frame=False, poses_by_frame={})
            poses_by_frame[frame.frame_index] = pose
        return TrackingCoordinateContext(use_world_frame=bool(frames), poses_by_frame=poses_by_frame)

    def _group_measurements(
        self,
        frame: TrackingFrameInput,
        config: TrackingConfig,
        coordinate_context: TrackingCoordinateContext,
    ) -> dict[str, list[TrackingMeasurement]]:
        grouped: dict[str, list[TrackingMeasurement]] = {}
        pose = coordinate_context.pose_for_frame(frame.frame_index)
        for annotation in frame.annotations:
            if annotation.source_model == "manual" or annotation.source_stage in {"interpolated", "predicted"}:
                continue
            if config.input_score_threshold is not None and annotation.score < config.input_score_threshold:
                continue
            measurement = self._measurement_from_annotation(annotation, pose)
            grouped.setdefault(annotation.class_name, []).append(measurement)
        return grouped

    def _associate_tracks(
        self,
        tracks: list[TrackState],
        measurements: list[TrackingMeasurement],
        config: TrackingConfig,
    ) -> list[tuple[int, int]]:
        if not tracks or not measurements:
            return []

        costs = np.full((len(tracks), len(measurements)), np.inf, dtype=np.float64)
        for track_index, track in enumerate(tracks):
            predicted_center = track.filter.box_state()[:3]
            predicted_size = track.filter.box_state()[3:6]
            for detection_index, measurement in enumerate(measurements):
                distance = float(np.linalg.norm(predicted_center - measurement.center))
                if distance > config.association_distance_m:
                    continue
                size_delta = self._size_delta(predicted_size, measurement.size)
                costs[track_index, detection_index] = distance + 0.25 * size_delta

        if not np.isfinite(costs).any():
            return []

        safe_costs = np.where(np.isfinite(costs), costs, config.association_distance_m + 1e6)
        row_indexes, column_indexes = linear_sum_assignment(safe_costs)
        matches: list[tuple[int, int]] = []
        for row_index, column_index in zip(row_indexes, column_indexes, strict=False):
            if not np.isfinite(costs[row_index, column_index]):
                continue
            matches.append((int(row_index), int(column_index)))
        return matches

    def _create_track(
        self,
        *,
        track_number: int,
        measurement: TrackingMeasurement,
        timestamp: float,
    ) -> TrackState:
        return TrackState(
            track_id=f"trk-{track_number:05d}",
            class_name=measurement.annotation.class_name,
            filter=KalmanBoxFilter.from_observation(measurement.observation),
            yaw=measurement.yaw,
            last_timestamp=timestamp,
            state_timestamp=timestamp,
            last_annotation=measurement.annotation,
            missed_frames=0,
        )

    def _predict_track_to_frame(self, track: TrackState, timestamp: float, config: TrackingConfig) -> None:
        if track.state_timestamp == timestamp:
            return
        delta_t = max(float(timestamp) - float(track.state_timestamp), 0.0)
        track.filter.predict(delta_t, config)
        track.state_timestamp = timestamp

    def _update_track(
        self,
        track: TrackState,
        measurement: TrackingMeasurement,
        timestamp: float,
        config: TrackingConfig,
    ) -> None:
        track.filter.update(measurement.observation, config)
        track.yaw = self._smooth_angle(track.yaw, measurement.yaw, config.yaw_smoothing_alpha)
        track.last_timestamp = timestamp
        track.state_timestamp = timestamp
        track.last_annotation = measurement.annotation
        track.missed_frames = 0

    def _measurement_from_annotation(self, annotation: TrackingBox3D, pose: np.ndarray | None) -> TrackingMeasurement:
        center = np.asarray([annotation.center.x, annotation.center.y, annotation.center.z], dtype=np.float64)
        size = np.asarray([annotation.size.x, annotation.size.y, annotation.size.z], dtype=np.float64)
        yaw = float(annotation.yaw)
        if pose is not None:
            center = self._transform_point(pose, center)
            yaw = self._transform_yaw(pose[:3, :3], yaw)
        return TrackingMeasurement(
            annotation=annotation,
            center=center,
            size=np.maximum(size, 0.01),
            yaw=self._normalize_angle(yaw),
        )

    def _annotation_from_track(
        self,
        *,
        track: TrackState,
        frame_index: int,
        pose: np.ndarray | None,
        source_annotation: TrackingBox3D,
        source_stage: Literal["detected", "predicted"],
    ) -> TrackingBox3D:
        state = track.filter.box_state()
        center = state[:3]
        size = np.maximum(state[3:6], 0.01)
        yaw = track.yaw
        if pose is not None:
            inverse_pose = np.linalg.inv(pose)
            center = self._transform_point(inverse_pose, center)
            yaw = self._transform_yaw(inverse_pose[:3, :3], yaw)

        annotation_id = source_annotation.id
        if source_stage == "predicted":
            annotation_id = f"{track.track_id}-pred-{frame_index:06d}"

        return source_annotation.model_copy(
            update={
                "id": annotation_id,
                "center": Vector3(x=float(center[0]), y=float(center[1]), z=float(center[2])),
                "size": Vector3(x=float(size[0]), y=float(size[1]), z=float(size[2])),
                "yaw": float(self._normalize_angle(yaw)),
                "coord_frame": "lidar",
                "track_id": track.track_id,
                "source_stage": source_stage,
            }
        )

    def _coerce_pose_matrix(self, ego_pose: list[list[float]] | None) -> np.ndarray | None:
        if ego_pose is None:
            return None
        try:
            matrix = np.asarray(ego_pose, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            return None
        return matrix

    def _transform_point(self, transform: np.ndarray, point: np.ndarray) -> np.ndarray:
        homogeneous = np.ones(4, dtype=np.float64)
        homogeneous[:3] = point
        transformed = transform @ homogeneous
        return transformed[:3]

    def _transform_yaw(self, rotation: np.ndarray, yaw: float) -> float:
        heading = np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
        transformed = rotation @ heading
        return math.atan2(float(transformed[1]), float(transformed[0]))

    def _smooth_angle(self, previous: float, current: float, previous_weight: float) -> float:
        delta = math.atan2(math.sin(current - previous), math.cos(current - previous))
        return self._normalize_angle(previous + (1.0 - previous_weight) * delta)

    def _normalize_angle(self, value: float) -> float:
        return math.atan2(math.sin(value), math.cos(value))

    def _size_delta(self, previous_size: np.ndarray, current_size: np.ndarray) -> float:
        baseline = np.maximum(np.asarray(previous_size, dtype=np.float64), 1e-3)
        return float(np.mean(np.abs(np.asarray(current_size, dtype=np.float64) - baseline) / baseline))

    def _ensure_not_cancelled(self, cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise TrackingCancelled("Tracking cancelled")


class TrackingEngine:
    def __init__(self) -> None:
        self._service = TrackingService()
        self._jobs: dict[str, WorkerTrackingJob] = {}
        self._lock = threading.Lock()

    def create_job(self, request: TrackingJobCreateRequest) -> TrackingJobStatusResponse:
        job = WorkerTrackingJob(job_id=str(uuid4()), request=request)
        with self._lock:
            self._jobs[job.job_id] = job
        thread = threading.Thread(target=self._run_job, args=(job,), name=f"tracking-{job.job_id}", daemon=True)
        thread.start()
        return job.snapshot()

    def get_status(self, job_id: str) -> TrackingJobStatusResponse:
        return self._get_job(job_id).snapshot()

    def cancel(self, job_id: str) -> TrackingJobStatusResponse:
        job = self._get_job(job_id)
        job.cancel_event.set()
        return job.snapshot()

    def get_result(self, job_id: str) -> TrackingJobResultResponse:
        job = self._get_job(job_id)
        with job.lock:
            if job.status != "completed" or job.result is None:
                raise RuntimeError(f"Tracking job is not completed: {job.status}")
            return job.result

    def _get_job(self, job_id: str) -> WorkerTrackingJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def _run_job(self, job: WorkerTrackingJob) -> None:
        try:
            job.mark_running()
            tracked = self._service.track(
                job.request.frames,
                job.request.config,
                progress_callback=job.update_progress,
                cancel_event=job.cancel_event,
            )
            results = [
                TrackingFrameResult(frame_index=frame.frame_index, annotations=tracked.get(frame.frame_index, []))
                for frame in job.request.frames
            ]
            job.mark_completed(
                TrackingJobResultResponse(
                    job_id=job.job_id,
                    method=job.request.config.method,
                    results=results,
                )
            )
        except TrackingCancelled:
            job.mark_cancelled()
        except Exception as exc:
            job.mark_failed(exc)


engine = TrackingEngine()
