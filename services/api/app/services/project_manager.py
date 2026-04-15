#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/api/app/services/project_manager.py
# @IDE     : vscode

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from app.adapters.base import sensor_match_get
from app.models import (
    CurrentProjectResponse,
    DatasetCapabilityReport,
    DatasetConfig,
    DatasetReference,
    DatasetSelectionRequest,
    DatasetValidationResult,
    FileBrowserEntry,
    ProjectCreateRequest,
    ProjectDatasetState,
    ProjectOpenRequest,
    ProjectState,
    ProjectSummary,
    ProjectTutorialState,
    ProjectUiState,
    ProjectUiStateUpdateRequest,
    SequenceSummary,
    TutorialStep,
)
from app.services.annotation_store import AnnotationStore
from app.services.dataset_registry import DatasetRegistry
from app.services.job_manager import JobManager
from app.utils import ensure_parent


class ProjectManager:
    WINDOWS_DRIVE_PATTERN = re.compile(r"^(?P<drive>[a-zA-Z]):[\\/]*(?P<rest>.*)$")

    def __init__(
        self,
        *,
        dataset_registry: DatasetRegistry,
        annotation_store: AnnotationStore,
        job_manager: JobManager,
        state_root: str,
        inactive_annotation_root: str,
        inactive_jobs_root: str,
        host_filesystem_root: str,
    ) -> None:
        self.dataset_registry = dataset_registry
        self.annotation_store = annotation_store
        self.job_manager = job_manager
        self.state_root = Path(state_root)
        self.inactive_annotation_root = Path(inactive_annotation_root)
        self.inactive_jobs_root = Path(inactive_jobs_root)
        self.host_filesystem_root = Path(host_filesystem_root)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.inactive_annotation_root.mkdir(parents=True, exist_ok=True)
        self.inactive_jobs_root.mkdir(parents=True, exist_ok=True)
        self._current_project_root: Path | None = None
        self._project_payload: dict | None = None

    @property
    def current_project_root(self) -> Path | None:
        return self._current_project_root

    async def restore_current_project(self) -> None:
        current_root = self._read_current_project_pointer()
        if current_root is None:
            await self._deactivate_services()
            return
        try:
            await self.open_project(ProjectOpenRequest(root_path=str(current_root)))
        except Exception:
            self._clear_current_project_pointer()
            await self._deactivate_services()

    async def create_project(self, payload: ProjectCreateRequest) -> ProjectState:
        root = self._normalize_path(payload.root_path)
        project_file = self._project_file(root)
        if project_file.exists():
            raise ValueError(f"Project already exists at {root}")

        (root / "annotations").mkdir(parents=True, exist_ok=True)
        (root / "cache" / "jobs").mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(parents=True, exist_ok=True)

        timestamp = self._now()
        project_payload = {
            "name": payload.name.strip() or root.name,
            "root_path": str(root),
            "created_at": timestamp,
            "updated_at": timestamp,
            "dataset": ProjectDatasetState().model_dump(),
            "ui_state": ProjectUiState().model_dump(),
            "tutorial": ProjectTutorialState().model_dump(),
        }
        self._write_project_payload(root, project_payload)
        return await self.open_project(ProjectOpenRequest(root_path=str(root)))

    async def open_project(self, payload: ProjectOpenRequest) -> ProjectState:
        root = self._normalize_path(payload.root_path)
        project_payload = self._read_project_payload(root)

        (root / "annotations").mkdir(parents=True, exist_ok=True)
        (root / "cache" / "jobs").mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(parents=True, exist_ok=True)

        self._current_project_root = root
        self._project_payload = project_payload

        self.annotation_store.set_root(str(root / "annotations"))
        self.dataset_registry.clear()
        await self.job_manager.set_jobs_root(str(root / "cache" / "jobs"))

        dataset_state = ProjectDatasetState.model_validate(project_payload.get("dataset") or {})
        if dataset_state.reference.dataset_id:
            try:
                dataset_state = await self._restore_project_dataset(dataset_state)
            except Exception as exc:
                dataset_state.validation = DatasetValidationResult(
                    ok=False,
                    message="Dataset validation failed while opening project",
                    errors=[str(exc)],
                )
                dataset_state.loaded_sequence_id = None
            project_payload["dataset"] = dataset_state.model_dump()

        self._write_current_project_pointer(root)
        self._save_project()
        return self.get_current_project()

    async def close_project(self) -> None:
        self._save_project()
        self._current_project_root = None
        self._project_payload = None
        self._clear_current_project_pointer()
        await self._deactivate_services()

    def get_current_project_response(self) -> CurrentProjectResponse:
        return CurrentProjectResponse(project=self.get_current_project())

    def get_current_project(self) -> ProjectState | None:
        if self._current_project_root is None or self._project_payload is None:
            return None

        dataset_state = ProjectDatasetState.model_validate(self._project_payload.get("dataset") or {})
        ui_state = ProjectUiState.model_validate(self._project_payload.get("ui_state") or {})
        sequences = self.dataset_registry.list_sequences()
        current_sequence_id = dataset_state.loaded_sequence_id
        if current_sequence_id and not any(sequence.id == current_sequence_id for sequence in sequences):
            current_sequence_id = sequences[0].id if sequences else None
        tutorial_state = self._build_tutorial_state(dataset_state)

        self._project_payload["tutorial"] = tutorial_state.model_dump()
        self._project_payload["dataset"] = dataset_state.model_dump()
        self._project_payload["ui_state"] = ui_state.model_dump()

        return ProjectState(
            project=ProjectSummary(
                name=self._project_payload["name"],
                root_path=self._project_payload["root_path"],
                created_at=self._project_payload["created_at"],
                updated_at=self._project_payload["updated_at"],
            ),
            dataset=dataset_state,
            tutorial=tutorial_state,
            ui_state=ui_state,
            sequences=sequences,
            current_sequence_id=current_sequence_id,
        )

    def list_filesystem_roots(self) -> list[FileBrowserEntry]:
        roots: list[Path] = []
        for candidate in self._host_drive_roots():
            if candidate.exists() and candidate not in roots:
                roots.append(candidate)
        for candidate in [Path.home(), Path("/workspace"), Path("/workspace-data"), Path("/")]:
            if candidate.exists() and candidate not in roots:
                roots.append(candidate)
        return [self._entry_from_path(root) for root in roots]

    def list_filesystem_entries(self, path: str) -> list[FileBrowserEntry]:
        target = self._normalize_path(path)
        if not target.exists():
            raise ValueError(f"Path not found: {target}")
        if not target.is_dir():
            raise ValueError(f"Path is not a directory: {target}")
        entries = []
        if target.parent != target and not self._is_host_drive_root(target):
            entries.append(self._entry_from_path(target.parent))
        children = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        entries.extend(self._entry_from_path(child) for child in children)
        return entries

    def validate_dataset(self, payload: DatasetSelectionRequest) -> ProjectState:
        self._require_active_project()
        reference, _, validation = self._validate_dataset_selection(payload)
        dataset_state = ProjectDatasetState.model_validate(self._project_payload.get("dataset") or {})
        dataset_state.reference = reference
        dataset_state.validation = validation
        dataset_state.loaded_sequence_id = None
        self._project_payload["dataset"] = dataset_state.model_dump()
        self._save_project()
        return self.get_current_project()

    async def load_dataset(self, payload: DatasetSelectionRequest) -> ProjectState:
        self._require_active_project()
        reference, config, validation = self._validate_dataset_selection(payload)
        if not validation.ok or config is None:
            raise ValueError(validation.errors[0] if validation.errors else validation.message)

        self.dataset_registry.clear()
        state = self.dataset_registry.load(config=config)
        summary = self.dataset_registry.get_summary(state.sequence_id)
        self.annotation_store.initialize_sequence(summary)
        await self.job_manager.resume_incomplete_jobs()

        dataset_state = ProjectDatasetState.model_validate(self._project_payload.get("dataset") or {})
        dataset_state.reference = reference
        dataset_state.validation = validation
        dataset_state.loaded_sequence_id = state.sequence_id
        self._project_payload["dataset"] = dataset_state.model_dump()
        self._save_project()
        return self.get_current_project()

    def update_ui_state(self, payload: ProjectUiStateUpdateRequest) -> ProjectState:
        self._require_active_project()
        current_state = ProjectUiState.model_validate(self._project_payload.get("ui_state") or {})
        updated_state = current_state.model_copy(update=payload.model_dump(exclude_unset=True))
        self._project_payload["ui_state"] = updated_state.model_dump()
        self._save_project()
        return self.get_current_project()

    async def _restore_project_dataset(self, dataset_state: ProjectDatasetState) -> ProjectDatasetState:
        last_reference = dataset_state.reference
        last_validation = dataset_state.validation
        for attempt in range(2):
            reference, config, validation = self._validate_dataset_selection(dataset_state.reference)
            dataset_state.reference = reference
            dataset_state.validation = validation
            last_reference = reference
            last_validation = validation
            if validation.ok and config is not None:
                state = self.dataset_registry.load(config=config)
                summary = self.dataset_registry.get_summary(state.sequence_id)
                self.annotation_store.initialize_sequence(summary)
                dataset_state.loaded_sequence_id = state.sequence_id
                await self.job_manager.resume_incomplete_jobs()
                return dataset_state
            dataset_state.loaded_sequence_id = None
            if attempt == 0:
                await asyncio.sleep(0.15)
        dataset_state.reference = last_reference
        dataset_state.validation = last_validation
        dataset_state.loaded_sequence_id = None
        return dataset_state

    def _validate_dataset_selection(
        self,
        payload: DatasetSelectionRequest | DatasetReference,
    ) -> tuple[DatasetReference, DatasetConfig | None, DatasetValidationResult]:
        try:
            reference = self._normalize_dataset_reference(payload)
        except Exception as exc:
            return (
                DatasetReference.model_validate(payload),
                None,
                DatasetValidationResult(ok=False, message="Dataset validation failed", errors=[str(exc)]),
            )

        try:
            config = self.dataset_registry.resolve_catalog_config(reference.dataset_id or "", reference.scene_id)
            adapter = self.dataset_registry.build_adapter(config)
            summary = self._build_sequence_summary(config, adapter)
            capability_report = self._build_capability_report(adapter)
            validation = DatasetValidationResult(
                ok=True,
                message=f"Validated dataset {summary.name}",
                sequence_summary=summary,
                capability_report=capability_report,
            )
            return reference, config, validation
        except Exception as exc:
            return (
                reference,
                None,
                DatasetValidationResult(ok=False, message="Dataset validation failed", errors=[str(exc)]),
            )

    async def _deactivate_services(self) -> None:
        self.dataset_registry.clear()
        self.annotation_store.set_root(str(self.inactive_annotation_root))
        await self.job_manager.set_jobs_root(str(self.inactive_jobs_root))

    def _build_sequence_summary(self, config: DatasetConfig, adapter) -> SequenceSummary:
        metadata = adapter.get_metadata()
        return SequenceSummary(
            id=self.dataset_registry.sequence_id_for_config(config),
            name=metadata.sequence_name,
            class_names=metadata.class_names,
            total_frames=len(adapter),
            fps=metadata.fps,
        )

    def _build_capability_report(self, adapter) -> DatasetCapabilityReport:
        lidar_available = False
        camera_available = False
        radar_available = False
        for index in range(len(adapter)):
            summary = adapter.get_frame_summary(index)
            lidar_available = lidar_available or bool(summary.get("lidar_path"))
            camera_available = camera_available or any(sensor_match_get(match, "available") for match in summary.get("camera", {}).values())
            radar_available = radar_available or any(sensor_match_get(match, "available") for match in summary.get("radar", {}).values())
            if lidar_available and camera_available and radar_available:
                break

        calibration = adapter.get_metadata().calibration or {}
        has_camera_intrinsics = bool(calibration.get("camera_rgb_intrinsics"))
        has_camera_transform = self._has_camera_transform(calibration)

        missing_requirements: list[str] = []
        if not lidar_available:
            missing_requirements.append("LiDAR data is missing, so 3D box annotation and auto-label are unavailable.")
        if not camera_available:
            missing_requirements.append("Camera data is missing, so camera visualization is unavailable.")
        if not radar_available:
            missing_requirements.append("Radar data is missing, so radar preview is unavailable.")
        if camera_available and not has_camera_intrinsics:
            missing_requirements.append("Camera intrinsics are missing, so camera projection is unavailable.")
        if camera_available and lidar_available and not has_camera_transform:
            missing_requirements.append("LiDAR/Camera transform is missing, so camera projection is unavailable.")

        return DatasetCapabilityReport(
            lidar_available=lidar_available,
            camera_available=camera_available,
            radar_available=radar_available,
            can_annotate_3d_box=lidar_available,
            can_project_to_camera=lidar_available and camera_available and has_camera_intrinsics and has_camera_transform,
            missing_requirements=missing_requirements,
        )

    def _has_camera_transform(self, calibration: dict) -> bool:
        for key, value in calibration.items():
            if key == "camera_rgb_intrinsics":
                continue
            if isinstance(value, list) and len(value) == 4 and all(isinstance(row, list) and len(row) == 4 for row in value):
                return True
        return False

    def _build_tutorial_state(self, dataset_state: ProjectDatasetState) -> ProjectTutorialState:
        has_reference = bool(dataset_state.reference.dataset_id)
        validation = dataset_state.validation
        validation_ok = bool(validation and validation.ok)
        capabilities = validation.capability_report if validation else None
        loaded = bool(dataset_state.loaded_sequence_id)
        reference_label = None
        if dataset_state.reference.dataset_id:
            reference_label = dataset_state.reference.dataset_id
            if dataset_state.reference.scene_id:
                reference_label = f"{reference_label} / {dataset_state.reference.scene_id}"

        steps = [
            TutorialStep(
                id="project",
                title="Create or open project",
                status="completed",
                description="Project workspace is ready.",
            ),
            TutorialStep(
                id="dataset-reference",
                title="Select dataset",
                status="completed" if has_reference else "pending",
                description="Choose one of the built-in dataset definitions.",
                detail=reference_label,
            ),
            TutorialStep(
                id="dataset-validate",
                title="Validate dataset",
                status="completed" if validation_ok else ("blocked" if validation else "pending"),
                description="Check that the dataset scene and adapter can be loaded.",
                detail=validation.message if validation else None,
            ),
            TutorialStep(
                id="dataset-load",
                title="Load sequence",
                status="completed" if loaded else ("blocked" if validation and not validation.ok else "pending"),
                description="Load the validated sequence into the workspace.",
                detail=dataset_state.loaded_sequence_id,
            ),
            TutorialStep(
                id="capabilities",
                title="Review capabilities",
                status="completed" if capabilities else "pending",
                description="Inspect which sensors and features are available.",
                detail="\n".join(capabilities.missing_requirements) if capabilities and capabilities.missing_requirements else None,
            ),
            TutorialStep(
                id="workspace",
                title="Enter visualize / annotate",
                status="completed" if loaded else "pending",
                description="Open the workspace and continue with visualization or annotation.",
            ),
        ]
        current_step = next((step.id for step in steps if step.status != "completed"), None)
        return ProjectTutorialState(current_step_id=current_step, steps=steps)

    def _normalize_dataset_reference(self, payload: DatasetSelectionRequest | DatasetReference) -> DatasetReference:
        raw_reference = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
        reference = DatasetReference.model_validate(raw_reference)
        if reference.dataset_id is not None:
            reference.dataset_id = reference.dataset_id.strip() or None
        if reference.scene_id is not None:
            reference.scene_id = reference.scene_id.strip() or None
        if not reference.dataset_id:
            raise ValueError("dataset_id is required")
        return reference

    def _save_project(self) -> None:
        if self._project_payload is None or self._current_project_root is None:
            return
        self._project_payload["updated_at"] = self._now()
        state = self.get_current_project()
        if state is not None:
            self._project_payload["tutorial"] = state.tutorial.model_dump()
            self._project_payload["dataset"] = state.dataset.model_dump()
            self._project_payload["ui_state"] = state.ui_state.model_dump()
        self._write_project_payload(self._current_project_root, self._project_payload)

    def _project_file(self, root: Path) -> Path:
        return root / "project.json"

    def _read_project_payload(self, root: Path) -> dict:
        project_file = self._project_file(root)
        if not project_file.exists():
            raise ValueError(f"Project file not found at {project_file}")
        payload = json.loads(project_file.read_text(encoding="utf-8"))
        payload.setdefault("root_path", str(root))
        payload.setdefault("dataset", ProjectDatasetState().model_dump())
        payload.setdefault("ui_state", ProjectUiState().model_dump())
        payload.setdefault("tutorial", ProjectTutorialState().model_dump())
        return payload

    def _write_project_payload(self, root: Path, payload: dict) -> None:
        project_file = self._project_file(root)
        ensure_parent(project_file)
        project_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_current_project_pointer(self, root: Path) -> None:
        pointer_path = self.state_root / "current_project.json"
        ensure_parent(pointer_path)
        pointer_path.write_text(json.dumps({"root_path": str(root)}, indent=2), encoding="utf-8")

    def _read_current_project_pointer(self) -> Path | None:
        pointer_path = self.state_root / "current_project.json"
        if not pointer_path.exists():
            return None
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
        root_path = payload.get("root_path")
        if not root_path:
            return None
        return self._normalize_path(root_path)

    def _clear_current_project_pointer(self) -> None:
        pointer_path = self.state_root / "current_project.json"
        if pointer_path.exists():
            pointer_path.unlink()

    def _normalize_path(self, value: str) -> Path:
        match = self.WINDOWS_DRIVE_PATTERN.match(value.strip())
        if match:
            drive_root = self.host_filesystem_root / match.group("drive").lower()
            remainder = match.group("rest").replace("\\", "/").lstrip("/")
            return (drive_root / remainder).resolve()
        return Path(value).expanduser().resolve()

    def _host_drive_roots(self) -> list[Path]:
        if not self.host_filesystem_root.exists():
            return []
        return sorted(
            [path for path in self.host_filesystem_root.iterdir() if path.is_dir()],
            key=lambda path: path.name.lower(),
        )

    def _is_host_drive_root(self, path: Path) -> bool:
        return path.parent == self.host_filesystem_root and path in self._host_drive_roots()

    def _now(self) -> str:
        return datetime.now(tz=timezone.utc).isoformat()

    def _require_active_project(self) -> Path:
        if self._current_project_root is None or self._project_payload is None:
            raise ValueError("No active project")
        return self._current_project_root

    def _entry_from_path(self, path: Path) -> FileBrowserEntry:
        name = path.name or str(path)
        if self._is_host_drive_root(path):
            name = f"{path.name.upper()}:"
        elif path == self.host_filesystem_root:
            name = "Host Filesystem"
        return FileBrowserEntry(
            name=name,
            path=str(path),
            is_dir=path.is_dir(),
            is_file=path.is_file(),
        )
