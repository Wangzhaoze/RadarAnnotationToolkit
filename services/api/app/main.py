#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/api/app/main.py
# @IDE     : vscode

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.dependencies import dataset_registry, project_manager
from app.routers.annotations import router as annotations_router
from app.routers.datasets import router as datasets_router
from app.routers.filesystem import router as filesystem_router
from app.routers.jobs import router as jobs_router
from app.routers.projects import router as projects_router

logger = logging.getLogger("radar-annotation-toolkit.api")

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3300,http://127.0.0.1:3300").split(",")
    if origin.strip()
]

app = FastAPI(title="Radar Annotation Toolkit API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["http://localhost:3300"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(datasets_router)
app.include_router(annotations_router)
app.include_router(jobs_router)
app.include_router(projects_router)
app.include_router(filesystem_router)

@app.get("/")
def root():
    return {"message": "Radar Annotation Toolkit API", "version": "0.1.0"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_sequences": len(dataset_registry.list_sequences()),
        "active_project": project_manager.current_project_root is not None,
    }


@app.on_event("startup")
async def restore_project_state():
    await project_manager.restore_current_project()
    if project_manager.current_project_root is None and not dataset_registry.list_sequences():
        logger.info("API started without loading a default dataset. Select a dataset scene from the File menu.")
