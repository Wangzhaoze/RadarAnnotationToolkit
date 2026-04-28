#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/api/app/routers/jobs.py
# @IDE     : vscode

from __future__ import annotations

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.dependencies import job_manager
from app.models import AutoLabelRequest, TrackingRequest

router = APIRouter(tags=["jobs"])


@router.post("/jobs/auto-label")
async def create_auto_label_job(payload: AutoLabelRequest):
    return await job_manager.create_auto_label_job(payload)


@router.post("/jobs/tracking")
async def create_tracking_job(payload: TrackingRequest):
    try:
        return await job_manager.create_tracking_job(payload)
    except Exception as exc:
        _raise_job_error(exc)


@router.get("/jobs/auto-label/models")
async def list_auto_label_models(sequence_id: str | None = None):
    try:
        return await job_manager.list_model_profiles(sequence_id=sequence_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/jobs/tracking/readiness")
def get_tracking_readiness(sequence_id: str, start_frame: int, end_frame: int):
    try:
        return job_manager.get_tracking_readiness(
            sequence_id=sequence_id,
            start_frame=start_frame,
            end_frame=end_frame,
        )
    except Exception as exc:
        _raise_job_error(exc)


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return job_manager.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _raise_job_error(exc: Exception) -> None:
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    raise exc


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    try:
        return await job_manager.pause_job(job_id)
    except Exception as exc:
        _raise_job_error(exc)


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    try:
        return await job_manager.resume_job(job_id)
    except Exception as exc:
        _raise_job_error(exc)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    try:
        return await job_manager.cancel_job(job_id)
    except Exception as exc:
        _raise_job_error(exc)


@router.websocket("/jobs/{job_id}/events")
async def job_events(websocket: WebSocket, job_id: str):
    await job_manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        job_manager.disconnect(job_id, websocket)
