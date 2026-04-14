from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.dependencies import project_manager

router = APIRouter(tags=["filesystem"])


@router.get("/filesystem/roots")
def get_filesystem_roots():
    try:
        return {"entries": project_manager.list_filesystem_roots()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/filesystem/entries")
def get_filesystem_entries(path: str = Query(...)):
    try:
        return {"entries": project_manager.list_filesystem_entries(path)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
