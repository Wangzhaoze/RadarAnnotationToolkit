#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/workers/base/worker_base/main.py
# @IDE     : vscode

from __future__ import annotations

import logging

from fastapi import HTTPException

import openpcdet.app.main as openpcdet_main
from tracking.service import (
    TrackingJobCreateRequest,
    TrackingJobResultResponse,
    TrackingJobStatusResponse,
    engine as tracking_engine,
)
from yolo.infer import YoloInferenceRequest, YoloInferenceResponse, engine as yolo_engine


logger = logging.getLogger("worker-base")

app = openpcdet_main.app
app.title = "Worker Base"
app.version = "0.3.0"

_original_catalog = openpcdet_main.engine.catalog


def _merged_catalog():
    catalog = _original_catalog()
    yolo_catalog = yolo_engine.catalog()
    catalog.providers = ["openpcdet", "yolo", "tracking"]
    catalog.provider_profiles = {
        "openpcdet": [profile.model_dump(mode="json") for profile in catalog.profiles],
        "yolo": [profile.model_dump(mode="json") for profile in yolo_catalog.profiles],
        "tracking": [
            {
                "id": "greedy_cv",
                "name": "Greedy Constant-Velocity Tracker",
                "framework": "worker-base",
                "description": "Associates 3D detections by class with XY constant-velocity prediction and greedy matching.",
                "is_available": True,
            }
        ],
    }
    return catalog


openpcdet_main.engine.catalog = _merged_catalog


@app.post("/infer/yolo", response_model=YoloInferenceResponse)
def infer_yolo(payload: YoloInferenceRequest):
    try:
        return yolo_engine.infer(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        logger.exception("YOLO inference failed for model_id=%s items=%s", payload.model_id, len(payload.items))
        raise HTTPException(status_code=500, detail=detail) from exc


@app.post("/tracking/jobs", response_model=TrackingJobStatusResponse)
def create_tracking_job(payload: TrackingJobCreateRequest):
    try:
        return tracking_engine.create_job(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        logger.exception("Tracking job creation failed")
        raise HTTPException(status_code=500, detail=detail) from exc


@app.get("/tracking/jobs/{job_id}", response_model=TrackingJobStatusResponse)
def get_tracking_job(job_id: str):
    try:
        return tracking_engine.get_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Tracking job not found: {job_id}") from exc


@app.post("/tracking/jobs/{job_id}/cancel", response_model=TrackingJobStatusResponse)
def cancel_tracking_job(job_id: str):
    try:
        return tracking_engine.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Tracking job not found: {job_id}") from exc


@app.get("/tracking/jobs/{job_id}/result", response_model=TrackingJobResultResponse)
def get_tracking_job_result(job_id: str):
    try:
        return tracking_engine.get_result(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Tracking job not found: {job_id}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
