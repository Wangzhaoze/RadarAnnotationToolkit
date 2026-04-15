#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/api/app/routers/projects.py
# @IDE     : vscode

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.dependencies import project_manager
from app.models import (
    CurrentProjectResponse,
    DatasetSelectionRequest,
    ProjectCreateRequest,
    ProjectOpenRequest,
    ProjectUiStateUpdateRequest,
)

router = APIRouter(tags=["projects"])


@router.get("/projects/current", response_model=CurrentProjectResponse)
def get_current_project():
    return project_manager.get_current_project_response()


@router.post("/projects/create")
async def create_project(payload: ProjectCreateRequest):
    try:
        return await project_manager.create_project(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/projects/open")
async def open_project(payload: ProjectOpenRequest):
    try:
        return await project_manager.open_project(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/projects/close")
async def close_project():
    await project_manager.close_project()
    return {"ok": True}


@router.patch("/projects/current/ui-state")
def update_project_ui_state(payload: ProjectUiStateUpdateRequest):
    try:
        return project_manager.update_ui_state(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/projects/current/dataset/validate")
def validate_dataset(payload: DatasetSelectionRequest):
    try:
        return project_manager.validate_dataset(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/projects/current/dataset/load")
async def load_dataset(payload: DatasetSelectionRequest):
    try:
        return await project_manager.load_dataset(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
