from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import WebSocket

from app.models import AutoLabelModelCatalog, AutoLabelRequest, Box3D, JobStatus, TrackingReadinessResponse, TrackingRequest
from app.services.annotation_store import AnnotationStore
from app.services.dataset_registry import DatasetRegistry
from app.services.tracking_service import FrameTrackingInput, TrackingCancelled, TrackingConfig, TrackingService

logger = logging.getLogger("radar-annotation-toolkit.jobs")


@dataclass
class JobControl:
    pause_requested: bool = False
    cancel_requested: bool = False


@dataclass
class TrackingAssociationProgress:
    completed_frames: int = 0
    total_frames: int = 0
    current_frame: int | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, completed_frames: int, total_frames: int, current_frame: int) -> None:
        with self._lock:
            self.completed_frames = completed_frames
            self.total_frames = total_frames
            self.current_frame = current_frame

    def snapshot(self) -> tuple[int, int, int | None]:
        with self._lock:
            return self.completed_frames, self.total_frames, self.current_frame


class JobManager:
    def __init__(
        self,
        worker_url: str,
        dataset_registry: DatasetRegistry,
        annotation_store: AnnotationStore,
        jobs_root: str,
    ) -> None:
        self.worker_url = worker_url.rstrip("/")
        self.dataset_registry = dataset_registry
        self.annotation_store = annotation_store
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, JobStatus] = {}
        self.requests: dict[str, AutoLabelRequest | TrackingRequest] = {}
        self.request_types: dict[str, str] = {}
        self.controls: dict[str, JobControl] = {}
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.connections: dict[str, list[WebSocket]] = {}
        self.worker_retries = 12
        self.tracking_service = TrackingService()
        self._load_jobs()

    async def set_jobs_root(self, jobs_root: str) -> None:
        await self._cancel_all_tasks()
        await self._close_connections()
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._reset_state()
        self._load_jobs()

    async def create_auto_label_job(self, request: AutoLabelRequest) -> JobStatus:
        self._validate_sequence_frame_range(request.sequence_id, request.start_frame, request.end_frame)
        total_frames = request.end_frame - request.start_frame + 1
        pending_frames, skipped_frames = self._build_frame_plan(request)
        next_frame = pending_frames[0] if pending_frames else None
        job = JobStatus(
            id=str(uuid4()),
            sequence_id=request.sequence_id,
            model_name=request.model_name,
            task_name=request.task_name,
            status="queued",
            start_frame=request.start_frame,
            end_frame=request.end_frame,
            total_frames=total_frames,
            skipped_frames=skipped_frames,
            processed_frames=skipped_frames,
            progress=round(skipped_frames / total_frames, 4) if total_frames else 1.0,
            current_frame=next_frame,
            message=self._queued_message(skipped_frames, next_frame, total_frames),
        )
        self.jobs[job.id] = job
        self.requests[job.id] = request
        self.request_types[job.id] = "auto_label"
        self.controls[job.id] = JobControl()
        self._persist_job(job.id)
        if not pending_frames:
            job.status = "completed"
            job.progress = 1.0
            job.message = f"All {total_frames} frames already have detection annotations"
            job.current_frame = None
            self._persist_job(job.id)
            return job
        self._start_task(job.id, request, pending_frames)
        return job

    async def create_tracking_job(self, request: TrackingRequest) -> JobStatus:
        self._validate_sequence_frame_range(request.sequence_id, request.start_frame, request.end_frame)
        readiness = self.get_tracking_readiness(
            sequence_id=request.sequence_id,
            start_frame=request.start_frame,
            end_frame=request.end_frame,
        )
        if not readiness.ready:
            raise ValueError(readiness.message)

        total_frames = request.end_frame - request.start_frame + 1
        job = JobStatus(
            id=str(uuid4()),
            sequence_id=request.sequence_id,
            model_name=self._tracking_model_name(request.method),
            task_name=request.task_name,
            status="queued",
            start_frame=request.start_frame,
            end_frame=request.end_frame,
            total_frames=total_frames,
            processed_frames=0,
            progress=0.0,
            stage=None,
            current_frame=request.start_frame,
            message=f"Queued {request.method} tracking for frames {request.start_frame}-{request.end_frame}",
        )
        self.jobs[job.id] = job
        self.requests[job.id] = request
        self.request_types[job.id] = "tracking"
        self.controls[job.id] = JobControl()
        self._persist_job(job.id)
        self._start_task(job.id, request)
        return job

    def get_tracking_readiness(self, *, sequence_id: str, start_frame: int, end_frame: int) -> TrackingReadinessResponse:
        self._validate_sequence_frame_range(sequence_id, start_frame, end_frame)
        total_frames = end_frame - start_frame + 1
        recorded_frames = 0
        next_missing_frame = None

        for frame_index in range(start_frame, end_frame + 1):
            if self.annotation_store.has_detection_record(sequence_id, frame_index):
                recorded_frames += 1
                continue
            if next_missing_frame is None:
                next_missing_frame = frame_index

        ready = recorded_frames == total_frames
        if ready:
            message = f"Tracking is available. Detection annotations exist for all {total_frames} frames."
        else:
            message = (
                f"Tracking is locked until frame-wise detection finishes for the whole range. "
                f"{recorded_frames}/{total_frames} frames are ready."
            )

        return TrackingReadinessResponse(
            ready=ready,
            annotated_frames=recorded_frames,
            total_frames=total_frames,
            next_missing_frame=next_missing_frame,
            message=message,
        )

    async def list_model_profiles(self) -> AutoLabelModelCatalog:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
            response = await self._request_worker(client, "GET", "/models")
            return AutoLabelModelCatalog.model_validate(response.json())

    async def resume_incomplete_jobs(self) -> None:
        for job_id, job in list(self.jobs.items()):
            if job.status not in {"queued", "running", "paused"}:
                continue
            request = self.requests.get(job_id)
            request_type = self.request_types.get(job_id, "auto_label")
            if request is None:
                job.status = "failed"
                job.error = "Missing auto-label request payload"
                job.message = "Auto-labeling failed"
                self._persist_job(job_id)
                continue
            if request_type == "tracking":
                try:
                    readiness = self.get_tracking_readiness(
                        sequence_id=request.sequence_id,
                        start_frame=request.start_frame,
                        end_frame=request.end_frame,
                    )
                except Exception as exc:
                    job.status = "failed"
                    job.error = self._format_error(exc)
                    job.current_frame = None
                    job.message = "Tracking failed to resume"
                    self._persist_job(job_id)
                    continue
                if not readiness.ready:
                    job.status = "failed"
                    job.error = readiness.message
                    job.current_frame = None
                    job.message = "Tracking failed to resume"
                    self._persist_job(job_id)
                    continue
                job.status = "queued"
                job.processed_frames = 0
                job.progress = 0.0
                job.stage = None
                job.current_frame = request.start_frame
                job.pause_requested = False
                job.cancel_requested = False
                job.message = f"Resuming tracking after restart from frame {request.start_frame}"
                self._persist_job(job_id)
                self._start_task(job_id, request)
                continue
            try:
                pending_frames = self._reconcile_job_progress(job, request)
            except Exception as exc:
                job.status = "failed"
                job.error = self._format_error(exc)
                job.current_frame = None
                job.message = "Auto-labeling failed to resume"
                job.pause_requested = False
                job.cancel_requested = False
                self._persist_job(job_id)
                continue
            control = self.controls.setdefault(job_id, JobControl())
            control.cancel_requested = False
            if not pending_frames:
                job.status = "completed"
                job.progress = 1.0
                job.current_frame = None
                job.pause_requested = False
                job.cancel_requested = False
                job.message = f"All {job.total_frames} frames already have detection annotations"
                self._persist_job(job_id)
                continue
            if job.status == "paused" or job.pause_requested:
                control.pause_requested = True
                job.status = "paused"
                job.pause_requested = True
                job.cancel_requested = False
                job.message = f"Paused. Ready to resume from frame {pending_frames[0]}"
                self._persist_job(job_id)
                continue
            control.pause_requested = False
            job.status = "queued"
            job.pause_requested = False
            job.cancel_requested = False
            job.message = f"Resuming after restart from frame {pending_frames[0]}"
            self._persist_job(job_id)
            self._start_task(job_id, request, pending_frames)

    async def pause_job(self, job_id: str) -> JobStatus:
        job = self.jobs[job_id]
        if self.request_types.get(job_id) == "tracking":
            raise ValueError("Tracking jobs do not support pause/resume. Stop the job and run it again if needed.")
        control = self.controls[job_id]
        if job.status in {"completed", "failed", "cancelled"}:
            raise ValueError(f"Job {job_id} is already finished")
        if control.pause_requested:
            return job
        control.pause_requested = True
        job.pause_requested = True
        if job.status == "queued":
            job.status = "paused"
            job.message = "Paused before inference started"
        else:
            job.message = "Pause requested. Waiting for the current frame to finish."
        await self._broadcast(job)
        return job

    async def resume_job(self, job_id: str) -> JobStatus:
        job = self.jobs[job_id]
        if self.request_types.get(job_id) == "tracking":
            raise ValueError("Tracking jobs do not support pause/resume. Run tracking again after it stops.")
        control = self.controls[job_id]
        if job.status in {"completed", "failed", "cancelled"}:
            raise ValueError(f"Job {job_id} is already finished")
        task = self.tasks.get(job_id)
        task_is_live = task is not None and not task.done()
        if not control.pause_requested and job.status != "paused" and task_is_live:
            return job
        control.pause_requested = False
        job.pause_requested = False
        pending_frames = self._reconcile_job_progress(job, self.requests[job_id])
        if not pending_frames:
            job.status = "completed"
            job.progress = 1.0
            job.current_frame = None
            job.message = f"All {job.total_frames} frames are already annotated"
            await self._broadcast(job)
            return job
        if not task_is_live:
            job.status = "queued"
            job.current_frame = pending_frames[0]
            job.message = f"Resuming from frame {pending_frames[0]}"
            await self._broadcast(job)
            self._start_task(job_id, self.requests[job_id], pending_frames)
            return job
        if job.status == "paused":
            job.status = "running"
        job.message = "Resuming auto-labeling"
        await self._broadcast(job)
        return job

    async def cancel_job(self, job_id: str) -> JobStatus:
        job = self.jobs[job_id]
        control = self.controls[job_id]
        if job.status in {"completed", "failed", "cancelled"}:
            return job
        control.cancel_requested = True
        job.cancel_requested = True
        is_tracking_job = self.request_types.get(job_id) == "tracking"
        if job.status in {"queued", "paused"}:
            control.cancel_requested = False
            control.pause_requested = False
            job.cancel_requested = False
            job.pause_requested = False
            job.status = "cancelled"
            job.stage = None
            job.message = self._stopped_message(job)
            await self._broadcast(job)
            return job
        if is_tracking_job:
            control.pause_requested = False
            job.pause_requested = False
            job.message = "Stop requested. Waiting for the current tracking stage to finish."
            await self._broadcast(job)
            return job
        control.pause_requested = False
        job.pause_requested = False
        job.status = "cancelled"
        job.stage = None
        job.current_frame = None
        job.cancel_requested = False
        job.message = self._stopping_message(job)
        task = self.tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        await self._broadcast(job)
        return job

    async def _process_job(self, job_id: str, request: AutoLabelRequest, pending_frames: list[int]) -> None:
        job = self.jobs[job_id]
        control = self.controls[job_id]
        if await self._finish_if_cancelled(job, control):
            return
        if control.pause_requested:
            job.status = "paused"
            job.message = "Paused before inference started"
        else:
            job.status = "running"
            if pending_frames:
                job.current_frame = pending_frames[0]
                if job.skipped_frames:
                    job.message = f"Found {job.skipped_frames} processed frames. Starting from frame {pending_frames[0]}"
                else:
                    job.message = f"Preparing {request.model_name} inference"
            else:
                job.message = f"Preparing {request.model_name} inference"
        await self._broadcast(job)
        sequence = self.dataset_registry.get(request.sequence_id)
        score_threshold = request.score_threshold
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            try:
                for frame_index in pending_frames:
                    if await self._wait_if_paused(job, control):
                        return
                    if await self._finish_if_cancelled(job, control):
                        return
                    job.status = "running"
                    job.current_frame = frame_index
                    warmup_note = " Worker/model warm-up can make the first frame slower." if job.processed_frames == 0 else ""
                    job.message = (
                        f"Running worker inference for frame {frame_index} "
                        f"({job.processed_frames + 1}/{job.total_frames}).{warmup_note}"
                    )
                    await self._broadcast(job)
                    frame_summary = sequence.adapter.get_frame_summary(frame_index)
                    response = await self._request_worker(
                        client,
                        "POST",
                        "/infer/openpcdet",
                        job=job,
                        json={
                            "model_profile_id": request.model_profile_id,
                            "model_name": request.model_name,
                            "model_cfg_file": request.model_cfg_file,
                            "ckpt_path": request.ckpt_path,
                            "class_names": sequence.config.class_names,
                            "items": [
                                {
                                    "frame_index": frame_index,
                                    "pointcloud_path": frame_summary["lidar_path"],
                                }
                            ],
                        },
                    )
                    payload = response.json()
                    annotations = [Box3D.model_validate(item) for item in payload["results"][0]["annotations"]]
                    if score_threshold is not None:
                        annotations = [annotation for annotation in annotations if annotation.score >= score_threshold]
                    self.annotation_store.replace_annotations(
                        request.sequence_id,
                        frame_index,
                        annotations,
                        detection_completed=True,
                    )
                    job.processed_frames += 1
                    job.progress = round(job.processed_frames / job.total_frames, 4)
                    job.message = f"Processed frame {frame_index} ({job.processed_frames}/{job.total_frames})"
                    await self._broadcast(job)
                    if await self._finish_if_cancelled(job, control):
                        return
            except asyncio.CancelledError:
                if job.status != "cancelled":
                    job.status = "cancelled"
                    job.message = "Auto-labeling stopped"
                    job.current_frame = None
                    job.pause_requested = False
                    job.cancel_requested = False
                    await self._broadcast(job)
                return
            except Exception as exc:
                job.status = "failed"
                job.error = self._format_error(exc)
                job.current_frame = None
                job.message = "Auto-labeling failed"
                await self._broadcast(job)
                return
            finally:
                self.tasks.pop(job_id, None)
        job.status = "completed"
        job.progress = 1.0
        job.current_frame = None
        job.pause_requested = False
        job.cancel_requested = False
        job.message = "Auto-labeling completed"
        await self._broadcast(job)

    async def _process_tracking_job(self, job_id: str, request: TrackingRequest) -> None:
        job = self.jobs[job_id]
        control = self.controls[job_id]
        sequence = self.dataset_registry.get(request.sequence_id)
        try:
            readiness = self.get_tracking_readiness(
                sequence_id=request.sequence_id,
                start_frame=request.start_frame,
                end_frame=request.end_frame,
            )
            if not readiness.ready:
                raise ValueError(readiness.message)

            job.status = "running"
            job.stage = "preparing_inputs"
            job.processed_frames = 0
            job.progress = 0.0
            job.current_frame = request.start_frame
            job.message = f"Preparing tracking inputs for frames {request.start_frame}-{request.end_frame}"
            await self._broadcast(job)

            tracking_config = TrackingConfig(
                method=request.method,
                input_score_threshold=request.input_score_threshold,
                max_gap_frames=request.max_gap_frames,
                max_track_lost_frames=request.max_track_lost_frames,
                match_distance_xy=request.match_distance_xy,
                match_z_delta=request.match_z_delta,
            )
            frame_inputs: list[FrameTrackingInput] = []
            for offset, frame_index in enumerate(range(request.start_frame, request.end_frame + 1), start=1):
                if await self._finish_if_cancelled(job, control):
                    return
                frame_summary = sequence.adapter.get_frame_summary(frame_index)
                frame_inputs.append(
                    FrameTrackingInput(
                        frame_index=frame_index,
                        timestamp=float(frame_summary["timestamp"]),
                        annotations=self.annotation_store.get_annotations(request.sequence_id, frame_index),
                    )
                )
                job.processed_frames = offset
                job.progress = self._tracking_progress("preparing_inputs", offset, job.total_frames)
                job.current_frame = frame_index
                job.message = f"Preparing tracking inputs ({offset}/{job.total_frames})"
                await self._broadcast(job)

            if await self._finish_if_cancelled(job, control):
                return

            association_progress = TrackingAssociationProgress(total_frames=job.total_frames)
            cancel_event = threading.Event()
            association_task = asyncio.create_task(
                asyncio.to_thread(
                    self.tracking_service.track,
                    frame_inputs,
                    tracking_config,
                    progress_callback=association_progress.update,
                    cancel_event=cancel_event,
                )
            )
            try:
                job.stage = "associating_tracks"
                job.processed_frames = 0
                job.progress = self._tracking_progress("associating_tracks", 0, job.total_frames)
                job.current_frame = request.start_frame
                job.message = f"Associating tracks (0/{job.total_frames})"
                await self._broadcast(job)
                last_snapshot: tuple[int, int, int | None] | None = (0, 0, None)
                association_poll_interval = 0.05
                while not association_task.done():
                    if control.cancel_requested:
                        cancel_event.set()
                    snapshot = association_progress.snapshot()
                    if snapshot == last_snapshot:
                        await asyncio.sleep(association_poll_interval)
                        continue
                    last_snapshot = snapshot
                    completed_frames, total_frames, current_frame = snapshot
                    total_frames = total_frames or job.total_frames
                    job.stage = "associating_tracks"
                    job.processed_frames = completed_frames
                    job.progress = self._tracking_progress("associating_tracks", completed_frames, total_frames)
                    job.current_frame = current_frame if current_frame is not None else request.start_frame
                    job.message = f"Associating tracks ({completed_frames}/{job.total_frames})"
                    await self._broadcast(job)
                    await asyncio.sleep(association_poll_interval)
                snapshot = association_progress.snapshot()
                if snapshot != last_snapshot:
                    completed_frames, total_frames, current_frame = snapshot
                    total_frames = total_frames or job.total_frames
                    job.stage = "associating_tracks"
                    job.processed_frames = completed_frames
                    job.progress = self._tracking_progress("associating_tracks", completed_frames, total_frames)
                    job.current_frame = current_frame if current_frame is not None else request.start_frame
                    job.message = f"Associating tracks ({completed_frames}/{job.total_frames})"
                    await self._broadcast(job)
                tracked_annotations = await association_task
            except TrackingCancelled:
                if await self._finish_if_cancelled(job, control):
                    return
                raise

            if await self._finish_if_cancelled(job, control):
                return

            job.stage = "writing_results"
            job.processed_frames = 0
            job.progress = self._tracking_progress("writing_results", 0, job.total_frames)
            job.message = "Applying tracked annotations"
            await self._broadcast(job)
            for offset, frame_index in enumerate(range(request.start_frame, request.end_frame + 1), start=1):
                if await self._finish_if_cancelled(job, control):
                    return
                existing_annotations = self.annotation_store.get_annotations(request.sequence_id, frame_index)
                next_annotations = self._merge_tracking_annotations(
                    existing_annotations=existing_annotations,
                    tracked_annotations=tracked_annotations.get(frame_index, []),
                )
                self.annotation_store.replace_annotations(request.sequence_id, frame_index, next_annotations)
                job.processed_frames = offset
                job.progress = self._tracking_progress("writing_results", offset, job.total_frames)
                job.current_frame = frame_index
                job.message = f"Writing tracking results ({offset}/{job.total_frames})"
                await self._broadcast(job)
        except asyncio.CancelledError:
            if job.status != "cancelled":
                job.status = "cancelled"
                job.stage = None
                job.message = "Tracking stopped"
                job.current_frame = None
                job.pause_requested = False
                job.cancel_requested = False
                await self._broadcast(job)
            return
        except Exception as exc:
            job.status = "failed"
            job.stage = None
            job.error = self._format_error(exc)
            job.current_frame = None
            job.message = "Tracking failed"
            await self._broadcast(job)
            return
        finally:
            self.tasks.pop(job_id, None)

        job.status = "completed"
        job.progress = 1.0
        job.stage = None
        job.processed_frames = job.total_frames
        job.current_frame = None
        job.pause_requested = False
        job.cancel_requested = False
        job.message = "Tracking completed"
        await self._broadcast(job)

    def get_job(self, job_id: str) -> JobStatus:
        return self.jobs[job_id]

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.setdefault(job_id, []).append(websocket)
        if job_id in self.jobs:
            await websocket.send_json(self.jobs[job_id].model_dump())

    def disconnect(self, job_id: str, websocket: WebSocket) -> None:
        if job_id in self.connections:
            self.connections[job_id] = [conn for conn in self.connections[job_id] if conn is not websocket]

    async def _wait_if_paused(self, job: JobStatus, control: JobControl) -> bool:
        while control.pause_requested and not control.cancel_requested:
            job.pause_requested = True
            if job.status != "paused":
                job.status = "paused"
                job.message = "Paused"
                await self._broadcast(job)
            await asyncio.sleep(0.25)
        if control.cancel_requested:
            return await self._finish_if_cancelled(job, control)
        if job.status == "paused":
            job.status = "running"
            job.pause_requested = False
            job.message = "Resuming auto-labeling"
            await self._broadcast(job)
        return False

    async def _finish_if_cancelled(self, job: JobStatus, control: JobControl) -> bool:
        if not control.cancel_requested:
            return False
        control.cancel_requested = False
        control.pause_requested = False
        job.status = "cancelled"
        job.stage = None
        job.cancel_requested = False
        job.pause_requested = False
        job.current_frame = None
        job.message = self._stopped_message(job)
        await self._broadcast(job)
        return True

    async def _broadcast(self, job: JobStatus) -> None:
        self._persist_job(job.id)
        for websocket in list(self.connections.get(job.id, [])):
            try:
                await websocket.send_json(job.model_dump())
            except Exception:
                self.disconnect(job.id, websocket)

    async def _request_worker(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        *,
        job: JobStatus | None = None,
        **kwargs,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(1, self.worker_retries + 1):
            started_at = time.perf_counter()
            try:
                response = await client.request(method, f"{self.worker_url}{path}", **kwargs)
                response.raise_for_status()
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                logger.info("Worker request %s %s completed in %.1f ms", method, path, elapsed_ms)
                return response
            except httpx.RequestError as exc:
                last_error = exc
                if job is not None:
                    job.message = f"Worker unavailable, retrying ({attempt}/{self.worker_retries})"
                    await self._broadcast(job)
                if attempt >= self.worker_retries:
                    break
                await asyncio.sleep(min(2 * attempt, 8))
        if last_error is not None:
            raise RuntimeError(
                f"Worker request failed after {self.worker_retries} attempts: {self._format_error(last_error)}"
            ) from last_error
        raise RuntimeError("Worker request failed without an exception")

    def _format_error(self, exc: Exception) -> str:
        message = str(exc).strip()
        if message:
            return message
        return exc.__class__.__name__

    def _tracking_progress(self, stage: str, completed_frames: int, total_frames: int) -> float:
        if total_frames <= 0:
            return 1.0
        clamped_completed = max(0, min(completed_frames, total_frames))
        stage_offsets = {
            "preparing_inputs": 0,
            "associating_tracks": total_frames,
            "writing_results": total_frames * 2,
        }
        completed_units = stage_offsets.get(stage, 0) + clamped_completed
        return round(completed_units / (total_frames * 3), 4)

    def _tracking_model_name(self, method: str) -> str:
        if method == "greedy_cv":
            return "Greedy CV Tracker"
        return method

    def _merge_tracking_annotations(
        self,
        *,
        existing_annotations: list[Box3D],
        tracked_annotations: list[Box3D],
    ) -> list[Box3D]:
        manual_annotations = [annotation for annotation in existing_annotations if annotation.source_model == "manual"]
        tracked_detected = {
            annotation.id: annotation
            for annotation in tracked_annotations
            if annotation.source_stage != "interpolated"
        }
        interpolated_annotations = [
            annotation
            for annotation in tracked_annotations
            if annotation.source_stage == "interpolated"
        ]
        next_detected: list[Box3D] = []
        for annotation in existing_annotations:
            if annotation.source_model == "manual" or annotation.source_stage == "interpolated":
                continue
            updated_annotation = tracked_detected.get(annotation.id)
            if updated_annotation is not None:
                next_detected.append(updated_annotation)
                continue
            next_detected.append(
                annotation.model_copy(
                    update={
                        "track_id": None,
                        "source_stage": "detected",
                    }
                )
            )
        return manual_annotations + next_detected + interpolated_annotations

    def _stopping_message(self, job: JobStatus) -> str:
        return "Stopping tracking" if job.task_name == "3D Tracking" else "Stopping auto-labeling"

    def _stopped_message(self, job: JobStatus) -> str:
        return "Tracking stopped" if job.task_name == "3D Tracking" else "Auto-labeling stopped"

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_root / f"{job_id}.json"

    def _load_jobs(self) -> None:
        for job_file in sorted(self.jobs_root.glob("*.json")):
            try:
                payload = json.loads(job_file.read_text(encoding="utf-8"))
                job = JobStatus.model_validate(payload["job"])
                request_type = payload.get("request_type", "auto_label")
                if request_type == "tracking":
                    request = TrackingRequest.model_validate(payload["request"])
                else:
                    request = AutoLabelRequest.model_validate(payload["request"])
            except Exception as exc:
                logger.warning("Failed to load persisted job from %s: %s", job_file, exc)
                continue
            self.jobs[job.id] = job
            self.requests[job.id] = request
            self.request_types[job.id] = request_type
            self.controls[job.id] = JobControl(
                pause_requested=job.pause_requested,
                cancel_requested=job.cancel_requested,
            )

    async def _cancel_all_tasks(self) -> None:
        live_tasks = [task for task in self.tasks.values() if not task.done()]
        if not live_tasks:
            self.tasks.clear()
            return
        for task in live_tasks:
            task.cancel()
        await asyncio.gather(*live_tasks, return_exceptions=True)
        self.tasks.clear()

    async def _close_connections(self) -> None:
        for sockets in list(self.connections.values()):
            for websocket in list(sockets):
                try:
                    await websocket.close(code=1001)
                except Exception:
                    pass
        self.connections.clear()

    def _reset_state(self) -> None:
        self.jobs.clear()
        self.requests.clear()
        self.request_types.clear()
        self.controls.clear()

    def _persist_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        request = self.requests.get(job_id)
        if job is None or request is None:
            return
        path = self._job_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "job": job.model_dump(),
                    "request_type": self.request_types.get(job_id, "auto_label"),
                    "request": request.model_dump(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _start_task(
        self,
        job_id: str,
        request: AutoLabelRequest | TrackingRequest,
        pending_frames: list[int] | None = None,
    ) -> None:
        request_type = self.request_types.get(job_id, "auto_label")
        if request_type == "tracking":
            self.tasks[job_id] = asyncio.create_task(self._process_tracking_job(job_id, request))
            return
        if pending_frames is None:
            raise ValueError("pending_frames is required for auto-label jobs")
        self.tasks[job_id] = asyncio.create_task(self._process_job(job_id, request, pending_frames))

    def _reconcile_job_progress(self, job: JobStatus, request: AutoLabelRequest) -> list[int]:
        pending_frames, _ = self._build_frame_plan(request)
        annotated_frames = job.total_frames - len(pending_frames)
        job.skipped_frames = min(job.skipped_frames, annotated_frames)
        job.processed_frames = max(job.skipped_frames, annotated_frames)
        job.progress = round(job.processed_frames / job.total_frames, 4) if job.total_frames else 1.0
        job.current_frame = pending_frames[0] if pending_frames else None
        return pending_frames

    def _build_frame_plan(self, request: AutoLabelRequest) -> tuple[list[int], int]:
        pending_frames: list[int] = []
        skipped_frames = 0
        for frame_index in range(request.start_frame, request.end_frame + 1):
            if self.annotation_store.has_detection_record(request.sequence_id, frame_index):
                skipped_frames += 1
                continue
            pending_frames.append(frame_index)
        return pending_frames, skipped_frames

    def _validate_sequence_frame_range(self, sequence_id: str, start_frame: int, end_frame: int) -> None:
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError("Invalid frame range")
        sequence = self.dataset_registry.get(sequence_id)
        total_frames = len(sequence.adapter)
        if end_frame >= total_frames:
            raise ValueError(f"Frame range exceeds sequence length ({total_frames} frames)")

    def _queued_message(self, skipped_frames: int, next_frame: int | None, total_frames: int) -> str:
        if skipped_frames and next_frame is not None:
            return f"Queued. Reusing {skipped_frames} processed frames and starting from frame {next_frame}"
        if skipped_frames == total_frames:
            return f"All {total_frames} frames already have detection annotations"
        return "Queued"
