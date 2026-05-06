#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-18
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/workers/base/yolo/infer_seg.py
# @IDE     : vscode

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


YOLO_SOURCE_ROOT = Path(os.getenv("YOLO_SOURCE_ROOT", "/workspace/services/workers/base/yolo"))
DEFAULT_YOLO_CKPT_ROOT = YOLO_SOURCE_ROOT.parent / "ckpts" / "yolo"
YOLO_CKPT_ROOT = Path(
    os.getenv(
        "YOLO_CKPT_ROOT",
        os.getenv("YOLO_ASSETS_ROOT", str(DEFAULT_YOLO_CKPT_ROOT)),
    )
)
DEFAULT_YOLO_MODEL_ID = "yolo26l_2d_detection"
YOLO_CLASS_NAMES = ["Vehicle", "Pedestrian", "Bicycle", "truck", "bus"]


class YoloInferenceItem(BaseModel):
    frame_index: int
    image_path: str


class YoloInferenceRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str | None = None
    items: list[YoloInferenceItem]


class YoloProfile(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    name: str
    framework: str
    task: str
    weight_name: str
    description: str
    ckpt_path: str
    class_names: list[str] = Field(default_factory=lambda: list(YOLO_CLASS_NAMES))
    default_score_threshold: float = 0.1
    is_available: bool


class BBox2D(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class YoloAnnotation(BaseModel):
    id: str
    class_name: str
    score: float
    bbox_2d: BBox2D
    mask_polygons: list[list[list[float]]] = Field(default_factory=list)
    source_model: str


class YoloInferenceResult(BaseModel):
    frame_index: int
    annotations: list[YoloAnnotation]


class YoloInferenceResponse(BaseModel):
    provider: str = "yolo"
    model_id: str
    model_name: str
    task: str
    results: list[YoloInferenceResult]


class YoloCatalogResponse(BaseModel):
    provider: str = "yolo"
    default_model_id: str
    profiles: list[YoloProfile]


YOLO_PROFILES = {
    "yolo26l_2d_detection": {
        "id": "yolo26l_2d_detection",
        "name": "YOLO26 Large 2D Object Detection",
        "framework": "Ultralytics",
        "task": "2d_object_detection",
        "weight_name": "yolo26l.pt",
        "description": "YOLO detection model for camera 2D bounding boxes.",
        "ckpt_path": str(YOLO_CKPT_ROOT / "yolo26l.pt"),
    },
    "yolo26l_2d_segmentation": {
        "id": "yolo26l_2d_segmentation",
        "name": "YOLO26 Large 2D Semantic Segmentation",
        "framework": "Ultralytics",
        "task": "2d_semantic_segmentation",
        "weight_name": "yolo26l-seg.pt",
        "description": "YOLO segmentation model for camera semantic masks.",
        "ckpt_path": str(YOLO_CKPT_ROOT / "yolo26l-seg.pt"),
    },
    "yolo26l_seg": {
        "id": "yolo26l_seg",
        "name": "YOLO26 Large Segmentation",
        "framework": "Ultralytics",
        "task": "instance_segmentation",
        "weight_name": "yolo26l-seg.pt",
        "description": "Compatibility profile for direct YOLO segmentation smoke tests.",
        "ckpt_path": str(YOLO_CKPT_ROOT / "yolo26l-seg.pt"),
    },
}


class YoloEngine:
    def catalog(self) -> YoloCatalogResponse:
        profiles = [
            YoloProfile(
                **profile,
                is_available=Path(profile["ckpt_path"]).exists(),
            )
            for profile in YOLO_PROFILES.values()
        ]
        return YoloCatalogResponse(default_model_id=DEFAULT_YOLO_MODEL_ID, profiles=profiles)

    def _resolve_profile(self, model_id: str | None) -> dict:
        resolved_model_id = model_id or DEFAULT_YOLO_MODEL_ID
        profile = YOLO_PROFILES.get(resolved_model_id)
        if profile is None:
            raise ValueError(f"Unknown YOLO model profile: {resolved_model_id}")
        return profile

    @lru_cache(maxsize=3)
    def _load_model(self, ckpt_path: str):
        from ultralytics import YOLO

        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"YOLO checkpoint not found: {ckpt}")
        return YOLO(str(ckpt))

    def _serialize_annotation(self, result, frame_index: int, annotation_index: int, model_name: str) -> YoloAnnotation:
        box = result.boxes[annotation_index]
        names = result.names or {}
        cls_index = int(box.cls.item()) if box.cls is not None else -1
        class_name = names.get(cls_index, str(cls_index))
        xyxy = box.xyxy[0].tolist()

        mask_polygons: list[list[list[float]]] = []
        masks = getattr(result, "masks", None)
        if masks is not None and getattr(masks, "xy", None) is not None and annotation_index < len(masks.xy):
            polygon = masks.xy[annotation_index]
            mask_polygons = [
                [[float(point[0]), float(point[1])] for point in polygon.tolist()]
            ]

        return YoloAnnotation(
            id=f"{frame_index}-{annotation_index}",
            class_name=class_name,
            score=float(box.conf.item()),
            bbox_2d=BBox2D(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
            ),
            mask_polygons=mask_polygons,
            source_model=model_name,
        )

    def infer(self, request: YoloInferenceRequest) -> YoloInferenceResponse:
        profile = self._resolve_profile(request.model_id)
        model_name = profile["name"]
        model = self._load_model(profile["ckpt_path"])

        results: list[YoloInferenceResult] = []
        for item in request.items:
            image_path = Path(item.image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {item.image_path}")

            predictions = model.predict(source=str(image_path), verbose=False, retina_masks=True)
            if not predictions:
                results.append(YoloInferenceResult(frame_index=item.frame_index, annotations=[]))
                continue

            prediction = predictions[0]
            annotations: list[YoloAnnotation] = []
            if prediction.boxes is not None:
                for annotation_index in range(len(prediction.boxes)):
                    annotations.append(
                        self._serialize_annotation(
                            prediction,
                            frame_index=item.frame_index,
                            annotation_index=annotation_index,
                            model_name=model_name,
                        )
                    )

            results.append(YoloInferenceResult(frame_index=item.frame_index, annotations=annotations))

        return YoloInferenceResponse(
            model_id=profile["id"],
            model_name=model_name,
            task=profile["task"],
            results=results,
        )


engine = YoloEngine()
