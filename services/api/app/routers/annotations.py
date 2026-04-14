from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.dependencies import annotation_store, dataset_registry
from app.models import CreateAnnotationRequest, SaveSequenceResponse, UpdateAnnotationRequest

router = APIRouter(tags=["annotations"])


@router.post("/annotations/{sequence_id}/frames/{frame_index}")
def create_annotation(sequence_id: str, frame_index: int, payload: CreateAnnotationRequest):
    try:
        dataset_registry.get(sequence_id)
        return annotation_store.create_annotation(sequence_id, frame_index, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/annotations/{sequence_id}/{frame_index}/{annotation_id}")
def update_annotation(sequence_id: str, frame_index: int, annotation_id: str, payload: UpdateAnnotationRequest):
    try:
        dataset_registry.get(sequence_id)
        return annotation_store.upsert_annotation(sequence_id, frame_index, annotation_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/annotations/{sequence_id}/{frame_index}/{annotation_id}")
def delete_annotation(sequence_id: str, frame_index: int, annotation_id: str):
    try:
        dataset_registry.get(sequence_id)
        annotation_store.delete_annotation(sequence_id, frame_index, annotation_id)
        return {"ok": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/annotations/{sequence_id}/save")
def save_sequence(sequence_id: str):
    try:
        summary = dataset_registry.get_summary(sequence_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Sequence {sequence_id} not found") from exc
    manifest_path, frame_paths = annotation_store.save_sequence(summary)
    return SaveSequenceResponse(sequence_id=sequence_id, manifest_path=manifest_path, frame_paths=frame_paths)
