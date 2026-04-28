#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/api/app/routers/datasets.py
# @IDE     : vscode

from __future__ import annotations

import mimetypes
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from app.adapters.base import sensor_match_get
from app.dependencies import annotation_store, dataset_registry, project_manager
from app.models import (
    DatasetSelectionRequest,
    FrameAssetUrls,
    FrameDetail,
    FrameLayerAvailability,
    LoadDatasetRequest,
    SemanticAssetUrls,
    SemanticAvailability,
    SensorLayerAvailability,
)
from app.utils import float_heatmap_to_png_bytes, radelft_radar_semantic_to_png_bytes

router = APIRouter(tags=["datasets"])
logger = logging.getLogger("radar-annotation-toolkit.api.datasets")
IMAGE_PREVIEW_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@router.get("/datasets")
def list_datasets():
    return {"sequences": dataset_registry.list_sequences()}


@router.get("/datasets/catalog")
def get_dataset_catalog():
    try:
        return dataset_registry.list_dataset_catalog()
    except Exception as exc:
        logger.exception("Dataset load failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/datasets/load")
async def load_dataset(payload: LoadDatasetRequest):
    try:
        if not payload.dataset_id:
            raise ValueError("dataset_id is required")
        if project_manager.current_project_root is not None:
            return await project_manager.load_dataset(
                DatasetSelectionRequest(dataset_id=payload.dataset_id, scene_id=payload.scene_id)
            )
        state = dataset_registry.load(config=dataset_registry.resolve_catalog_config(payload.dataset_id, payload.scene_id))
        summary = dataset_registry.get_summary(state.sequence_id)
        annotation_store.initialize_sequence(summary)
        return {"sequence": summary, "ramap_cache": state.adapter.get_ramap_cache_status()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/sequences/{sequence_id}/timeline")
def get_timeline(sequence_id: str):
    try:
        return dataset_registry.get_timeline(sequence_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _semantic_availability(adapter, frame_index: int, summary: dict) -> SemanticAvailability:
    raw = adapter.get_semantic_summary(frame_index) or {}
    camera = {
        name: bool((raw.get("camera") or {}).get(name))
        for name in summary.get("camera", {})
    }
    radar = {
        name: bool((raw.get("radar") or {}).get(name))
        for name in summary.get("radar", {})
    }
    return SemanticAvailability(
        lidar=bool(raw.get("lidar")),
        camera=camera,
        radar=radar,
    )


def _merge_annotation_semantics(sequence_id: str, frame_index: int, availability: SemanticAvailability) -> SemanticAvailability:
    camera = dict(availability.camera)
    for name in camera:
        if annotation_store.has_camera_semantic(sequence_id, frame_index, name):
            camera[name] = True
    return SemanticAvailability(
        lidar=availability.lidar,
        camera=camera,
        radar=availability.radar,
    )


def _radar_preview_urls(adapter, sequence_id: str, frame_index: int, summary: dict) -> dict[str, str | None]:
    urls: dict[str, str | None] = {}
    for name in summary["radar"]:
        preview_path = adapter.get_radar_preview_path(frame_index, name)
        urls[name] = f"/sequences/{sequence_id}/frames/{frame_index}/radar/{name}/preview" if preview_path else None
    return urls


def _sensor_layer_availability(
    *,
    intensity_available: bool,
    semantic_available: bool,
    default_mode: str | None,
    intensity_reason: str | None,
    semantic_reason: str | None,
) -> SensorLayerAvailability:
    resolved_default = default_mode if default_mode in {"intensity", "semantic"} else None
    if resolved_default == "intensity" and not intensity_available:
        resolved_default = None
    if resolved_default == "semantic" and not semantic_available:
        resolved_default = None
    if resolved_default is None:
        if intensity_available:
            resolved_default = "intensity"
        elif semantic_available:
            resolved_default = "semantic"
    return SensorLayerAvailability(
        intensity=intensity_available,
        semantic=semantic_available,
        default_mode=resolved_default,
        intensity_reason=None if intensity_available else intensity_reason,
        semantic_reason=None if semantic_available else semantic_reason,
    )


def _layer_availability(adapter, frame_index: int, summary: dict, semantic_available: SemanticAvailability) -> FrameLayerAvailability:
    defaults = adapter.get_layer_defaults(frame_index) or {}
    lidar_intensity = bool(summary.get("lidar_path"))
    lidar_semantic = semantic_available.lidar
    camera_layers = {}
    for name, match in summary.get("camera", {}).items():
        camera_layers[name] = _sensor_layer_availability(
            intensity_available=bool(sensor_match_get(match, "available") and sensor_match_get(match, "path")),
            semantic_available=bool(semantic_available.camera.get(name)),
            default_mode=(defaults.get("camera") or {}).get(name),
            intensity_reason="No camera frame for this sample",
            semantic_reason="No semantic mask for this frame",
        )
    radar_layers = {}
    for name in summary.get("radar", {}):
        cache_status = adapter.get_ramap_cache_status() or {}
        cache_running = bool(cache_status.get("running"))
        cache_ready = bool(cache_status.get("intensity_available"))
        intensity_reason = "RAMap intensity cache is being prepared" if cache_running else "RAMap intensity cache is not available"
        if cache_ready:
            intensity_reason = "No RAMap intensity for this frame"
        radar_layers[name] = _sensor_layer_availability(
            intensity_available=bool(adapter.get_radar_preview_path(frame_index, name)),
            semantic_available=bool(semantic_available.radar.get(name)),
            default_mode=(defaults.get("radar") or {}).get(name),
            intensity_reason=intensity_reason,
            semantic_reason="No semantic radar layer for this frame",
        )
    return FrameLayerAvailability(
        lidar=_sensor_layer_availability(
            intensity_available=lidar_intensity,
            semantic_available=lidar_semantic,
            default_mode=defaults.get("lidar"),
            intensity_reason="No LiDAR frame for this sample",
            semantic_reason="No LiDAR semantic labels for this frame",
        ),
        camera=camera_layers,
        radar=radar_layers,
    )


@router.get("/sequences/{sequence_id}/frames/{frame_index}")
def get_frame(sequence_id: str, frame_index: int):
    try:
        state = dataset_registry.get(sequence_id)
        summary = state.adapter.get_frame_summary(frame_index)
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    semantic_available = _merge_annotation_semantics(
        sequence_id,
        frame_index,
        _semantic_availability(state.adapter, frame_index, summary),
    )
    layer_availability = _layer_availability(state.adapter, frame_index, summary, semantic_available)
    return FrameDetail(
        sequence_id=sequence_id,
        frame_index=frame_index,
        timestamp=summary["timestamp"],
        annotations=annotation_store.get_annotations(sequence_id, frame_index),
        annotations_2d=annotation_store.get_annotations_2d(sequence_id, frame_index),
        asset_urls=FrameAssetUrls(
            lidar=(f"/sequences/{sequence_id}/frames/{frame_index}/lidar" if summary.get("lidar_path") else None),
            camera={
                name: (
                    f"/sequences/{sequence_id}/frames/{frame_index}/camera/{name}"
                    if sensor_match_get(match, "available") and sensor_match_get(match, "path")
                    else None
                )
                for name, match in summary["camera"].items()
            },
            radar_preview=_radar_preview_urls(state.adapter, sequence_id, frame_index, summary),
            semantic=SemanticAssetUrls(
                lidar=(
                    f"/sequences/{sequence_id}/frames/{frame_index}/lidar/semantic"
                    if semantic_available.lidar
                    else None
                ),
                camera={
                    name: (
                        f"/sequences/{sequence_id}/frames/{frame_index}/camera/{name}/semantic"
                        if semantic_available.camera.get(name)
                        else None
                    )
                    for name in summary["camera"]
                },
                radar={
                    name: (
                        f"/sequences/{sequence_id}/frames/{frame_index}/radar/{name}/semantic"
                        if semantic_available.radar.get(name)
                        else None
                    )
                    for name in summary["radar"]
                },
            ),
        ),
        semantic_available=semantic_available,
        layer_availability=layer_availability,
        calibration=state.adapter.get_metadata().calibration,
        camera=summary["camera"],
        radar=summary["radar"],
    )


@router.get("/sequences/{sequence_id}/frames/{frame_index}/lidar")
def get_lidar_asset(sequence_id: str, frame_index: int):
    try:
        state = dataset_registry.get(sequence_id)
        summary = state.adapter.get_frame_summary(frame_index)
        if not summary.get("lidar_path"):
            raise HTTPException(status_code=404, detail="LiDAR frame not available")
        frame = state.adapter.get_frame(frame_index)
        payload = np.asarray(frame.lidar_points, dtype=np.float32).tobytes()
        return Response(content=payload, media_type="application/octet-stream")
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sequences/{sequence_id}/frames/{frame_index}/lidar/semantic")
def get_lidar_semantic_asset(sequence_id: str, frame_index: int):
    try:
        labels = dataset_registry.get(sequence_id).adapter.get_lidar_semantic_labels(frame_index)
        if labels is None:
            raise HTTPException(status_code=404, detail="LiDAR semantic labels not available")
        payload = np.asarray(labels, dtype=np.int32).reshape(-1).tobytes()
        return Response(content=payload, media_type="application/octet-stream")
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sequences/{sequence_id}/frames/{frame_index}/camera/{camera_name}")
def get_camera_asset(sequence_id: str, frame_index: int, camera_name: str):
    try:
        summary = dataset_registry.get(sequence_id).adapter.get_frame_summary(frame_index)
        match = summary["camera"][camera_name]
        path = sensor_match_get(match, "path")
        if not sensor_match_get(match, "available") or not path:
            raise HTTPException(status_code=404, detail="Camera frame not available")
        media_type, _ = mimetypes.guess_type(path)
        return FileResponse(path, media_type=media_type or "application/octet-stream")
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sequences/{sequence_id}/frames/{frame_index}/camera/{camera_name}/semantic")
def get_camera_semantic_asset(sequence_id: str, frame_index: int, camera_name: str):
    try:
        generated_path = annotation_store.get_camera_semantic_path(sequence_id, frame_index, camera_name)
        path = generated_path if generated_path.exists() else dataset_registry.get(sequence_id).adapter.get_camera_semantic_path(frame_index, camera_name)
        if not path:
            raise HTTPException(status_code=404, detail="Camera semantic frame not available")
        media_type, _ = mimetypes.guess_type(str(path))
        return FileResponse(path, media_type=media_type or "image/png")
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sequences/{sequence_id}/frames/{frame_index}/radar/{radar_name}/preview")
def get_radar_preview(sequence_id: str, frame_index: int, radar_name: str):
    try:
        adapter = dataset_registry.get(sequence_id).adapter
        path = adapter.get_radar_preview_path(frame_index, radar_name)
        if not path:
            raise HTTPException(status_code=404, detail="Radar frame not available")
        path = Path(path)
        if path.suffix.lower() in IMAGE_PREVIEW_EXTENSIONS:
            media_type, _ = mimetypes.guess_type(str(path))
            return FileResponse(path, media_type=media_type or "image/png")
        return Response(content=float_heatmap_to_png_bytes(path), media_type="image/png")
    except (KeyError, IndexError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sequences/{sequence_id}/ramap-cache")
def get_ramap_cache_status(sequence_id: str):
    try:
        status = dataset_registry.get(sequence_id).adapter.get_ramap_cache_status()
        if status is None:
            return {"available": False}
        return {"available": True, **status}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sequences/{sequence_id}/frames/{frame_index}/radar/{radar_name}/semantic")
def get_radar_semantic_asset(sequence_id: str, frame_index: int, radar_name: str):
    try:
        path = dataset_registry.get(sequence_id).adapter.get_radar_semantic_path(frame_index, radar_name)
        if not path:
            raise HTTPException(status_code=404, detail="Radar semantic frame not available")
        path = Path(path)
        if path.suffix.lower() == ".npy":
            return Response(content=radelft_radar_semantic_to_png_bytes(path), media_type="image/png")
        media_type, _ = mimetypes.guess_type(str(path))
        return FileResponse(path, media_type=media_type or "image/png")
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
