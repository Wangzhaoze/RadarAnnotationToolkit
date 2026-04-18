#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/api/app/services/annotation_store.py
# @IDE     : vscode

from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from app.models import Box2DAnnotation, Box3D, CreateAnnotationRequest, SequenceSummary, UpdateAnnotationRequest
from app.utils import ensure_parent


class AnnotationStore:
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, OrderedDict[int, list[Box3D]]] = {}
        self._known_frames: dict[str, set[int]] = {}
        self._max_cached_frames = 12

    def set_root(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache.clear()
        self._known_frames.clear()

    def _sequence_root(self, sequence_id: str) -> Path:
        return self.root / sequence_id

    def _manifest_path(self, sequence_id: str) -> Path:
        return self._sequence_root(sequence_id) / "manifest.json"

    def _frame_path(self, sequence_id: str, frame_index: int) -> Path:
        return self._sequence_root(sequence_id) / "frames" / f"{frame_index:06d}.json"

    def initialize_sequence(self, summary: SequenceSummary) -> None:
        self._cache.setdefault(summary.id, OrderedDict())
        self._known_frames.setdefault(summary.id, set())
        manifest_path = self._manifest_path(summary.id)
        self._index_sequence(summary.id)
        if not manifest_path.exists():
            self.save_sequence(summary)

    def _index_sequence(self, sequence_id: str) -> None:
        frame_dir = self._sequence_root(sequence_id) / "frames"
        known_frames: set[int] = set()
        if frame_dir.exists():
            for frame_file in frame_dir.glob("*.json"):
                try:
                    known_frames.add(int(frame_file.stem))
                except ValueError:
                    continue
        self._known_frames[sequence_id] = known_frames

    def get_annotations(self, sequence_id: str, frame_index: int) -> list[Box3D]:
        return list(self._read_frame(sequence_id, frame_index))

    def get_annotations_2d(self, sequence_id: str, frame_index: int) -> list[Box2DAnnotation]:
        payload = self._read_frame_payload(sequence_id, frame_index)
        if payload is None:
            return []
        return [Box2DAnnotation.model_validate(item) for item in payload.get("annotations_2d", [])]

    def has_annotations(self, sequence_id: str, frame_index: int) -> bool:
        return bool(self._read_frame(sequence_id, frame_index))

    def has_detection_record(self, sequence_id: str, frame_index: int) -> bool:
        payload = self._read_frame_payload(sequence_id, frame_index)
        if payload is None:
            return False
        meta = payload.get("meta")
        if isinstance(meta, dict) and "detection_completed" in meta:
            return bool(meta["detection_completed"])
        return True

    def has_2d_detection_record(self, sequence_id: str, frame_index: int) -> bool:
        return self._has_meta_flag(sequence_id, frame_index, "yolo_2d_detection_completed")

    def has_2d_segmentation_record(self, sequence_id: str, frame_index: int) -> bool:
        return self._has_meta_flag(sequence_id, frame_index, "yolo_2d_segmentation_completed")

    def _has_meta_flag(self, sequence_id: str, frame_index: int, key: str) -> bool:
        payload = self._read_frame_payload(sequence_id, frame_index)
        if payload is None:
            return False
        meta = payload.get("meta")
        return bool(meta.get(key)) if isinstance(meta, dict) else False

    def create_annotation(self, sequence_id: str, frame_index: int, payload: CreateAnnotationRequest) -> Box3D:
        annotation = Box3D(
            id=datetime.now(tz=timezone.utc).strftime("ann-%Y%m%d%H%M%S%f"),
            class_name=payload.class_name,
            score=payload.score,
            center=payload.center,
            size=payload.size,
            yaw=payload.yaw,
            source_model="manual",
            coord_frame="lidar",
        )
        annotations = self._read_frame(sequence_id, frame_index)
        annotations.append(annotation)
        self._write_frame(sequence_id, frame_index, annotations)
        return annotation

    def upsert_annotation(self, sequence_id: str, frame_index: int, annotation_id: str, payload: UpdateAnnotationRequest) -> Box3D:
        annotations = self._read_frame(sequence_id, frame_index)
        for idx, annotation in enumerate(annotations):
            if annotation.id == annotation_id:
                updated = annotation.model_copy(update=payload.model_dump(exclude_unset=True))
                annotations[idx] = updated
                self._write_frame(sequence_id, frame_index, annotations)
                return updated
        raise KeyError(f"Annotation {annotation_id} not found")

    def replace_annotations(
        self,
        sequence_id: str,
        frame_index: int,
        annotations: list[Box3D],
        *,
        detection_completed: bool | None = None,
    ) -> None:
        self._write_frame(
            sequence_id,
            frame_index,
            annotations,
            detection_completed=detection_completed,
        )

    def replace_annotations_2d(
        self,
        sequence_id: str,
        frame_index: int,
        annotations_2d: list[Box2DAnnotation],
        *,
        detection_completed: bool | None = None,
    ) -> None:
        payload = self._read_frame_payload(sequence_id, frame_index) or {}
        annotations = [Box3D.model_validate(item) for item in payload.get("annotations", [])]
        meta = dict(payload.get("meta") or {})
        if detection_completed is not None:
            meta["yolo_2d_detection_completed"] = detection_completed
        self._write_frame_payload(
            sequence_id,
            frame_index,
            annotations=annotations,
            annotations_2d=annotations_2d,
            meta=meta,
        )

    def write_camera_semantic_mask(
        self,
        sequence_id: str,
        frame_index: int,
        camera_name: str,
        image_bytes: bytes,
    ) -> str:
        path = self.get_camera_semantic_path(sequence_id, frame_index, camera_name)
        ensure_parent(path)
        path.write_bytes(image_bytes)
        payload = self._read_frame_payload(sequence_id, frame_index) or {}
        annotations = [Box3D.model_validate(item) for item in payload.get("annotations", [])]
        annotations_2d = [Box2DAnnotation.model_validate(item) for item in payload.get("annotations_2d", [])]
        meta = dict(payload.get("meta") or {})
        meta["yolo_2d_segmentation_completed"] = True
        self._write_frame_payload(
            sequence_id,
            frame_index,
            annotations=annotations,
            annotations_2d=annotations_2d,
            meta=meta,
        )
        return str(path)

    def get_camera_semantic_path(self, sequence_id: str, frame_index: int, camera_name: str = "rgb") -> Path:
        return self._sequence_root(sequence_id) / "camera_semantic" / camera_name / f"{frame_index:06d}.png"

    def has_camera_semantic(self, sequence_id: str, frame_index: int, camera_name: str = "rgb") -> bool:
        return self.get_camera_semantic_path(sequence_id, frame_index, camera_name).exists()

    def delete_annotation(self, sequence_id: str, frame_index: int, annotation_id: str) -> None:
        current = self._read_frame(sequence_id, frame_index)
        annotations = [annotation for annotation in current if annotation.id != annotation_id]
        self._write_frame(sequence_id, frame_index, annotations)

    def _read_frame(self, sequence_id: str, frame_index: int) -> list[Box3D]:
        sequence_cache = self._cache.setdefault(sequence_id, OrderedDict())
        if frame_index in sequence_cache:
            annotations = sequence_cache.pop(frame_index)
            sequence_cache[frame_index] = annotations
            return list(annotations)
        payload = self._read_frame_payload(sequence_id, frame_index)
        if payload is None:
            return []
        annotations = [Box3D.model_validate(item) for item in payload.get("annotations", [])]
        self._remember_frame(sequence_id, frame_index, annotations)
        return list(annotations)

    def _read_frame_payload(self, sequence_id: str, frame_index: int) -> dict | None:
        path = self._frame_path(sequence_id, frame_index)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _remember_frame(self, sequence_id: str, frame_index: int, annotations: list[Box3D]) -> None:
        sequence_cache = self._cache.setdefault(sequence_id, OrderedDict())
        if frame_index in sequence_cache:
            sequence_cache.pop(frame_index)
        sequence_cache[frame_index] = list(annotations)
        while len(sequence_cache) > self._max_cached_frames:
            sequence_cache.popitem(last=False)

    def _write_frame(
        self,
        sequence_id: str,
        frame_index: int,
        annotations: list[Box3D],
        *,
        detection_completed: bool | None = None,
    ) -> None:
        path = self._frame_path(sequence_id, frame_index)
        existing_payload = self._read_frame_payload(sequence_id, frame_index)
        annotations_2d = [
            Box2DAnnotation.model_validate(item)
            for item in (existing_payload or {}).get("annotations_2d", [])
        ]
        existing_meta = (existing_payload or {}).get("meta")
        meta = dict(existing_meta) if isinstance(existing_meta, dict) else {}
        if detection_completed is None:
            existing_meta = existing_payload.get("meta") if isinstance(existing_payload, dict) else None
            if isinstance(existing_meta, dict) and "detection_completed" in existing_meta:
                detection_completed = bool(existing_meta["detection_completed"])
            elif existing_payload is not None:
                detection_completed = True
            else:
                detection_completed = False
        meta["detection_completed"] = detection_completed
        self._write_frame_payload(
            sequence_id,
            frame_index,
            annotations=annotations,
            annotations_2d=annotations_2d,
            meta=meta,
        )

    def _write_frame_payload(
        self,
        sequence_id: str,
        frame_index: int,
        *,
        annotations: list[Box3D],
        annotations_2d: list[Box2DAnnotation],
        meta: dict,
    ) -> None:
        path = self._frame_path(sequence_id, frame_index)
        ensure_parent(path)
        meta.setdefault("detection_completed", False)
        payload = {
            "frame_index": frame_index,
            "annotations": [annotation.model_dump() for annotation in annotations],
            "annotations_2d": [annotation.model_dump() for annotation in annotations_2d],
            "meta": meta,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._known_frames.setdefault(sequence_id, set()).add(frame_index)
        self._remember_frame(sequence_id, frame_index, annotations)

    def save_sequence(self, summary: SequenceSummary) -> tuple[str, list[str]]:
        self._index_sequence(summary.id)
        frame_paths = [str(self._frame_path(summary.id, frame_index)) for frame_index in sorted(self._known_frames.setdefault(summary.id, set()))]

        manifest_path = self._manifest_path(summary.id)
        ensure_parent(manifest_path)
        manifest_path.write_text(
            json.dumps(
                {
                    "sequence_id": summary.id,
                    "sequence_name": summary.name,
                    "class_names": summary.class_names,
                    "total_frames": summary.total_frames,
                    "updated_at": datetime.now(tz=timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(manifest_path), frame_paths
