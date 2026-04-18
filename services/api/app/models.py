#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/api/app/models.py
# @IDE     : vscode

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class SensorMatch(BaseModel):
    index: int | None = None
    timestamp: float | None = None
    path: str | None = None
    delta_ms: float | None = None
    available: bool = False


class Box3D(BaseModel):
    id: str
    class_name: str
    score: float = 0.0
    center: Vector3
    size: Vector3
    yaw: float = 0.0
    source_model: str = "manual"
    coord_frame: Literal["lidar"] = "lidar"
    track_id: str | None = None
    source_stage: Literal["manual", "detected", "interpolated"] | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_tracking_defaults(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if payload.get("source_stage") is None:
            payload["source_stage"] = "manual" if payload.get("source_model") == "manual" else "detected"
        if "track_id" not in payload:
            payload["track_id"] = None
        return payload


class BBox2D(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Box2DAnnotation(BaseModel):
    id: str
    class_name: str
    score: float = 0.0
    bbox_2d: BBox2D
    mask_polygons: list[list[list[float]]] = Field(default_factory=list)
    source_model: str = "manual"
    source_stage: Literal["manual", "detected"] | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_source_stage(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if payload.get("source_stage") is None:
            payload["source_stage"] = "manual" if payload.get("source_model") == "manual" else "detected"
        return payload


class DatasetConfig(BaseModel):
    adapter: str
    dataset_root: str
    calib_root: str | None = None
    sequence_name: str
    class_names: list[str] = Field(default_factory=lambda: ["Car", "Pedestrian", "Cyclist"])
    camera_tolerance_ms: float = 75.0
    radar_tolerance_ms: float = 125.0
    options: dict[str, Any] = Field(default_factory=dict)


class LoadDatasetRequest(BaseModel):
    dataset_id: str | None = None
    scene_id: str | None = None


class DatasetReference(BaseModel):
    dataset_id: str | None = None
    scene_id: str | None = None


class SequenceSummary(BaseModel):
    id: str
    name: str
    class_names: list[str]
    total_frames: int
    fps: float


class TimelineFrame(BaseModel):
    frame_index: int
    timestamp: float
    lidar_path: str
    camera: dict[str, SensorMatch]
    radar: dict[str, SensorMatch]


class SequenceTimelineResponse(BaseModel):
    sequence: SequenceSummary
    frames: list[TimelineFrame]


class SemanticAssetUrls(BaseModel):
    lidar: str | None = None
    camera: dict[str, str | None] = Field(default_factory=dict)
    radar: dict[str, str | None] = Field(default_factory=dict)


class SemanticAvailability(BaseModel):
    lidar: bool = False
    camera: dict[str, bool] = Field(default_factory=dict)
    radar: dict[str, bool] = Field(default_factory=dict)


class SensorLayerAvailability(BaseModel):
    intensity: bool = False
    semantic: bool = False
    default_mode: Literal["intensity", "semantic"] | None = None
    intensity_reason: str | None = None
    semantic_reason: str | None = None


class FrameLayerAvailability(BaseModel):
    lidar: SensorLayerAvailability = Field(default_factory=SensorLayerAvailability)
    camera: dict[str, SensorLayerAvailability] = Field(default_factory=dict)
    radar: dict[str, SensorLayerAvailability] = Field(default_factory=dict)


class FrameAssetUrls(BaseModel):
    lidar: str | None
    camera: dict[str, str | None]
    radar_preview: dict[str, str | None]
    semantic: SemanticAssetUrls = Field(default_factory=SemanticAssetUrls)


class FrameDetail(BaseModel):
    sequence_id: str
    frame_index: int
    timestamp: float
    annotations: list[Box3D]
    annotations_2d: list[Box2DAnnotation] = Field(default_factory=list)
    asset_urls: FrameAssetUrls
    semantic_available: SemanticAvailability = Field(default_factory=SemanticAvailability)
    layer_availability: FrameLayerAvailability = Field(default_factory=FrameLayerAvailability)
    calibration: dict[str, Any]
    camera: dict[str, SensorMatch]
    radar: dict[str, SensorMatch]


class CreateAnnotationRequest(BaseModel):
    class_name: str = "Car"
    score: float = 0.0
    center: Vector3 = Field(default_factory=lambda: Vector3(x=0.0, y=0.0, z=0.0))
    size: Vector3 = Field(default_factory=lambda: Vector3(x=4.0, y=1.8, z=1.6))
    yaw: float = 0.0


class UpdateAnnotationRequest(BaseModel):
    class_name: str | None = None
    score: float | None = None
    center: Vector3 | None = None
    size: Vector3 | None = None
    yaw: float | None = None
    source_model: str | None = None


class TrackingRequest(BaseModel):
    sequence_id: str
    start_frame: int
    end_frame: int
    method: Literal["greedy_cv"] = "greedy_cv"
    input_score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    max_gap_frames: int = Field(default=2, ge=0)
    max_track_lost_frames: int = Field(default=2, ge=0)
    match_distance_xy: float | None = Field(default=None, gt=0.0)
    match_z_delta: float | None = Field(default=None, gt=0.0)
    task_name: str = "3D Tracking"


class AutoLabelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    sequence_id: str
    task_name: str = "3D Bounding Box"
    model_name: str = "OpenPCDet PV-RCNN"
    model_profile_id: str | None = None
    start_frame: int
    end_frame: int
    model_cfg_file: str | None = None
    ckpt_path: str | None = None
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class AutoLabelModelProfile(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    name: str
    architecture: str | None = None
    framework: str
    weight_name: str
    description: str
    cfg_file: str | None = None
    ckpt_path: str
    class_names: list[str] = Field(default_factory=list)
    default_score_threshold: float = 0.1
    is_available: bool
    provider: str = "openpcdet"
    task: str | None = None


class AutoLabelModelCatalog(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    default_model_profile_id: str
    profiles: list[AutoLabelModelProfile]


class JobStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    sequence_id: str
    model_name: str
    task_name: str
    status: Literal["queued", "running", "paused", "completed", "failed", "cancelled"]
    start_frame: int
    end_frame: int
    total_frames: int
    skipped_frames: int = 0
    processed_frames: int = 0
    progress: float = 0.0
    stage: Literal["preparing_inputs", "associating_tracks", "writing_results"] | None = None
    current_frame: int | None = None
    pause_requested: bool = False
    cancel_requested: bool = False
    message: str = ""
    error: str | None = None


class SaveSequenceResponse(BaseModel):
    sequence_id: str
    manifest_path: str
    frame_paths: list[str]


class TrackingReadinessResponse(BaseModel):
    ready: bool
    annotated_frames: int
    total_frames: int
    next_missing_frame: int | None = None
    message: str


class DatasetCapabilityReport(BaseModel):
    lidar_available: bool = False
    camera_available: bool = False
    radar_available: bool = False
    can_annotate_3d_box: bool = False
    can_project_to_camera: bool = False
    missing_requirements: list[str] = Field(default_factory=list)


class DatasetValidationResult(BaseModel):
    ok: bool
    message: str
    errors: list[str] = Field(default_factory=list)
    sequence_summary: SequenceSummary | None = None
    capability_report: DatasetCapabilityReport | None = None


class DatasetSceneOption(BaseModel):
    id: str
    name: str
    available: bool = True
    reason: str | None = None


class DatasetCatalogEntry(BaseModel):
    id: str
    name: str
    class_names: list[str] = Field(default_factory=list)
    default_scene_id: str | None = None
    scenes: list[DatasetSceneOption] = Field(default_factory=list)


class DatasetCatalogResponse(BaseModel):
    default_dataset_id: str | None = None
    default_scene_id: str | None = None
    datasets: list[DatasetCatalogEntry] = Field(default_factory=list)


class FileBrowserEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    is_file: bool


class ProjectSummary(BaseModel):
    name: str
    root_path: str
    created_at: str
    updated_at: str


class ProjectUiState(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    active_menu: str = "file"
    frame_index: int = 0
    selected_model_profile_id: str | None = None
    score_threshold: float = 0.1
    interaction_mode: str = "observe"
    transform_mode: str = "translate"
    view_mode: str = "3d"


class TutorialStep(BaseModel):
    id: str
    title: str
    status: Literal["pending", "completed", "blocked"]
    description: str
    detail: str | None = None


class ProjectTutorialState(BaseModel):
    current_step_id: str | None = None
    steps: list[TutorialStep] = Field(default_factory=list)


class ProjectDatasetState(BaseModel):
    reference: DatasetReference = Field(default_factory=DatasetReference)
    validation: DatasetValidationResult | None = None
    loaded_sequence_id: str | None = None


class ProjectState(BaseModel):
    project: ProjectSummary
    dataset: ProjectDatasetState = Field(default_factory=ProjectDatasetState)
    tutorial: ProjectTutorialState = Field(default_factory=ProjectTutorialState)
    ui_state: ProjectUiState = Field(default_factory=ProjectUiState)
    sequences: list[SequenceSummary] = Field(default_factory=list)
    current_sequence_id: str | None = None


class CurrentProjectResponse(BaseModel):
    project: ProjectState | None = None


class ProjectCreateRequest(BaseModel):
    name: str
    root_path: str


class ProjectOpenRequest(BaseModel):
    root_path: str


class ProjectUiStateUpdateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    active_menu: str | None = None
    frame_index: int | None = Field(default=None, ge=0)
    selected_model_profile_id: str | None = None
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    interaction_mode: str | None = None
    transform_mode: str | None = None
    view_mode: str | None = None


class DatasetSelectionRequest(BaseModel):
    dataset_id: str | None = None
    scene_id: str | None = None
