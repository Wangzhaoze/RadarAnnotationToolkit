#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/workers/base/openpcdet/app/main.py
# @IDE     : vscode

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
import logging
from numpy.lib.recfunctions import structured_to_unstructured
from pydantic import BaseModel, ConfigDict, Field

from app.model_profiles import DEFAULT_MODEL_PROFILE_ID, MODEL_PROFILES


logger = logging.getLogger("worker-base.openpcdet")


def _create_openpcdet_logger():
    from pcdet.utils import common_utils

    model_logger = logging.getLogger(common_utils.__name__)
    if model_logger.handlers:
        return model_logger
    return common_utils.create_logger()


class InferenceItem(BaseModel):
    frame_index: int
    pointcloud_path: str


class InferenceRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_profile_id: str | None = None
    model_name: str | None = None
    model_cfg_file: str | None = None
    ckpt_path: str | None = None
    class_names: list[str] = Field(default_factory=list)
    items: list[InferenceItem]


class WorkerBox3D(BaseModel):
    id: str
    class_name: str
    score: float
    center: dict[str, float]
    size: dict[str, float]
    yaw: float
    source_model: str
    coord_frame: Literal["lidar"] = "lidar"


class InferenceResult(BaseModel):
    frame_index: int
    annotations: list[WorkerBox3D]


class InferenceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_profile_id: str
    model_name: str
    results: list[InferenceResult]


class ModelProfile(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    name: str
    architecture: str
    framework: str
    weight_name: str
    description: str
    cfg_file: str
    ckpt_path: str
    class_names: list[str]
    preprocess_point_cloud_range: list[float] | None = None
    default_score_threshold: float = 0.1
    is_available: bool
    provider: str = "openpcdet"
    task: str | None = "3d_lidar_bbox"
    dataset_id: str | None = None
    training_dataset: str | None = None
    network_label: str | None = None
    config_label: str | None = None


class ModelCatalogResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    default_model_profile_id: str
    profiles: list[ModelProfile]
    providers: list[str] = Field(default_factory=lambda: ["openpcdet"])
    provider_profiles: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


def _normalize_points(
    points: np.ndarray,
    preprocess_point_cloud_range: list[float] | tuple[float, ...] | None = None,
    model_point_cloud_range: list[float] | tuple[float, ...] | None = None,
    target_num_point_features: int | None = None,
) -> np.ndarray:
    if getattr(points.dtype, "names", None):
        points = structured_to_unstructured(points)
    else:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2:
            raise ValueError("Expected point cloud with shape (N, 3+) or a structured array")

    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]

    if points.ndim != 2:
        raise ValueError("Expected point cloud with shape (N, 3+) or a structured array")
    required_features = max(4, int(target_num_point_features or 4))

    if points.shape[1] >= required_features:
        points = points[:, :required_features]
    elif points.shape[1] >= 4:
        points = points[:, :4]
    elif points.shape[1] == 3:
        intensity = np.linalg.norm(points[:, :3], axis=1, keepdims=True).astype(np.float32)
        points = np.concatenate([points, intensity], axis=1)
    else:
        raise ValueError("Expected point cloud with shape (N, 3+) or a structured array")

    if points.shape[1] < required_features:
        padding = np.zeros((points.shape[0], required_features - points.shape[1]), dtype=np.float32)
        points = np.concatenate([points, padding], axis=1)

    if preprocess_point_cloud_range is not None:
        x_min, y_min, z_min, x_max, y_max, z_max = [float(value) for value in preprocess_point_cloud_range]
        range_mask = (
            (points[:, 0] > x_min)
            & (points[:, 0] < x_max)
            & (points[:, 1] > y_min)
            & (points[:, 1] < y_max)
            & (points[:, 2] > z_min)
            & (points[:, 2] < z_max)
        )
        points = points[range_mask]

    if points.shape[0]:
        intensity = points[:, 3]
        intensity_min = float(intensity.min())
        intensity_max = float(intensity.max())
        if intensity_max > intensity_min:
            points[:, 3] = (intensity - intensity_min) / (intensity_max - intensity_min + 1e-6)

    if model_point_cloud_range is not None:
        x_min, y_min, z_min, x_max, y_max, z_max = [float(value) for value in model_point_cloud_range]
        range_mask = (
            (points[:, 0] >= x_min)
            & (points[:, 0] <= x_max)
            & (points[:, 1] >= y_min)
            & (points[:, 1] <= y_max)
            & (points[:, 2] >= z_min)
            & (points[:, 2] <= z_max)
        )
        points = points[range_mask]

    max_points = 120000
    if points.shape[0] > max_points:
        sample_indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[sample_indices]

    return points.astype(np.float32, copy=False)


def _load_points(
    pointcloud_path: str,
    preprocess_point_cloud_range: list[float] | tuple[float, ...] | None = None,
    model_point_cloud_range: list[float] | tuple[float, ...] | None = None,
    target_num_point_features: int | None = None,
) -> np.ndarray:
    path = Path(pointcloud_path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {pointcloud_path}")
    if path.suffix.lower() == ".bin":
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    elif path.suffix.lower() == ".npy":
        points = np.load(path)
    else:
        raise ValueError(f"Unsupported point cloud format: {path.suffix}")
    return _normalize_points(
        points,
        preprocess_point_cloud_range=preprocess_point_cloud_range,
        model_point_cloud_range=model_point_cloud_range,
        target_num_point_features=target_num_point_features,
    )


class OpenPCDetEngine:
    def __init__(self) -> None:
        self._inference_count = 0
        self._model_refresh_interval = max(0, int(os.getenv("MODEL_REFRESH_INTERVAL", "0")))

    def _resolve_profile(self, profile_id: str | None) -> dict:
        resolved_profile_id = profile_id or DEFAULT_MODEL_PROFILE_ID
        profile = MODEL_PROFILES.get(resolved_profile_id)
        if profile is None:
            raise ValueError(f"Unknown OpenPCDet model profile: {resolved_profile_id}")
        return profile

    def catalog(self) -> ModelCatalogResponse:
        profiles = []
        for profile in MODEL_PROFILES.values():
            cfg_file = Path(profile["cfg_file"])
            ckpt_path = Path(profile["ckpt_path"])
            profiles.append(
                ModelProfile(
                    **profile,
                    is_available=cfg_file.exists() and ckpt_path.exists(),
                )
            )
        return ModelCatalogResponse(default_model_profile_id=DEFAULT_MODEL_PROFILE_ID, profiles=profiles)

    @lru_cache(maxsize=4)
    def _load_model(self, cfg_file: str, ckpt_file: str):
        import copy

        import torch
        from pcdet.config import cfg as base_cfg
        from pcdet.config import cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network

        cfg = copy.deepcopy(base_cfg)
        cfg_from_yaml_file(cfg_file, cfg)
        model_logger = _create_openpcdet_logger()

        class ModelDataset(DatasetTemplate):
            def __len__(self):
                return 0

            def __getitem__(self, index):
                raise IndexError(index)

        dataset = ModelDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=False,
            root_path=None,
            logger=model_logger,
        )
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        use_cuda = torch.cuda.is_available()
        model.load_params_from_file(filename=ckpt_file, logger=model_logger, to_cpu=not use_cuda)
        model.to(torch.device("cuda:0" if use_cuda else "cpu"))
        model.eval()
        return model, cfg, use_cuda

    def _infer_openpcdet(
        self,
        points: np.ndarray,
        frame_index: int,
        model_name: str,
        cfg_file: str,
        ckpt_file: str,
        output_class_names: list[str] | None,
    ) -> list[WorkerBox3D]:
        import torch
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import load_data_to_gpu

        model, cfg, use_cuda = self._load_model(cfg_file, ckpt_file)

        class SingleFrameDataset(DatasetTemplate):
            def __init__(self, points_array: np.ndarray):
                super().__init__(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=None, logger=None)
                self.points_array = points_array

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return self.prepare_data(data_dict={"points": self.points_array, "frame_id": index})

        dataset = SingleFrameDataset(points)
        data_dict = dataset[0]
        data_dict = dataset.collate_batch([data_dict])
        if use_cuda:
            load_data_to_gpu(data_dict)

        with torch.no_grad():
            pred_dicts, _ = model.forward(data_dict)

        pred_boxes = pred_dicts[0]["pred_boxes"].detach().cpu().numpy()
        pred_scores = pred_dicts[0]["pred_scores"].detach().cpu().numpy()
        pred_labels = pred_dicts[0]["pred_labels"].detach().cpu().numpy()
        resolved_class_names = output_class_names if output_class_names and len(output_class_names) == len(cfg.CLASS_NAMES) else cfg.CLASS_NAMES

        results: list[WorkerBox3D] = []
        for idx, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels, strict=False)):
            label_index = int(label) - 1
            class_name = resolved_class_names[label_index] if 0 <= label_index < len(resolved_class_names) else "Unknown"
            results.append(
                WorkerBox3D(
                    id=f"{frame_index}-{idx}",
                    class_name=class_name,
                    score=float(score),
                    center={"x": float(box[0]), "y": float(box[1]), "z": float(box[2])},
                    size={"x": float(box[3]), "y": float(box[4]), "z": float(box[5])},
                    yaw=float(box[6]),
                    source_model=model_name,
                    coord_frame="lidar",
                )
            )
        self._inference_count += 1
        refresh_due = (
            use_cuda
            and self._model_refresh_interval > 0
            and self._inference_count % self._model_refresh_interval == 0
        )
        if use_cuda:
            if self._inference_count % 50 == 0:
                allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                logger.info(
                    "CUDA memory after %s frames: allocated=%.1f MiB reserved=%.1f MiB",
                    self._inference_count,
                    allocated_mb,
                    reserved_mb,
                )
            if refresh_due:
                logger.info(
                    "Refreshing cached OpenPCDet model after %s frames to reduce long-run CUDA/native instability",
                    self._inference_count,
                )
            del pred_dicts
            del data_dict
            del dataset
            if refresh_due:
                del model
                self._load_model.cache_clear()
            torch.cuda.empty_cache()
        return results

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        profile = self._resolve_profile(request.model_profile_id)
        cfg_file = request.model_cfg_file or profile["cfg_file"]
        ckpt_file = request.ckpt_path or profile["ckpt_path"]
        model_name = request.model_name or profile["name"]

        if not Path(cfg_file).exists():
            raise FileNotFoundError(f"OpenPCDet config file not found: {cfg_file}")
        if not Path(ckpt_file).exists():
            raise FileNotFoundError(f"OpenPCDet checkpoint not found: {ckpt_file}")

        _, cfg, _ = self._load_model(cfg_file, ckpt_file)
        preprocess_point_cloud_range = profile.get("preprocess_point_cloud_range")
        model_point_cloud_range = getattr(cfg.DATA_CONFIG, "POINT_CLOUD_RANGE", None)
        point_feature_encoding = getattr(cfg.DATA_CONFIG, "POINT_FEATURE_ENCODING", None)
        target_num_point_features = len(getattr(point_feature_encoding, "src_feature_list", []) or [])

        results: list[InferenceResult] = []
        for item in request.items:
            points = _load_points(
                item.pointcloud_path,
                preprocess_point_cloud_range=preprocess_point_cloud_range,
                model_point_cloud_range=model_point_cloud_range,
                target_num_point_features=target_num_point_features,
            )
            if points.size == 0:
                results.append(InferenceResult(frame_index=item.frame_index, annotations=[]))
                continue
            annotations = self._infer_openpcdet(
                points,
                frame_index=item.frame_index,
                model_name=model_name,
                cfg_file=cfg_file,
                ckpt_file=ckpt_file,
                output_class_names=request.class_names or profile["class_names"],
            )
            results.append(InferenceResult(frame_index=item.frame_index, annotations=annotations))
        return InferenceResponse(model_profile_id=profile["id"], model_name=model_name, results=results)


app = FastAPI(title="OpenPCDet Worker", version="0.2.0")
engine = OpenPCDetEngine()


@app.get("/")
def root():
    catalog = engine.catalog()
    return {
        "message": "Worker Base",
        "provider": "openpcdet",
        "default_model_profile_id": catalog.default_model_profile_id,
        "available_models": [profile.id for profile in catalog.profiles if profile.is_available],
    }


@app.get("/health")
def health():
    catalog = engine.catalog()
    return {
        "status": "ok",
        "provider": "openpcdet",
        "default_model_profile_id": catalog.default_model_profile_id,
        "available_models": [profile.id for profile in catalog.profiles if profile.is_available],
    }


@app.get("/models", response_model=ModelCatalogResponse)
def list_models():
    return engine.catalog()


@app.post("/infer/openpcdet", response_model=InferenceResponse)
def infer_openpcdet(payload: InferenceRequest):
    try:
        return engine.infer(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        logger.exception(
            "OpenPCDet inference failed for profile=%s model=%s items=%s",
            payload.model_profile_id,
            payload.model_name,
            len(payload.items),
        )
        raise HTTPException(status_code=500, detail=detail) from exc
