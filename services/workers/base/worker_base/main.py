from __future__ import annotations

import logging

from fastapi import HTTPException

import openpcdet.app.main as openpcdet_main
from yolo.infer import YoloInferenceRequest, YoloInferenceResponse, engine as yolo_engine


logger = logging.getLogger("worker-base")

app = openpcdet_main.app
app.title = "Worker Base"
app.version = "0.3.0"

_original_catalog = openpcdet_main.engine.catalog


def _merged_catalog():
    catalog = _original_catalog()
    yolo_catalog = yolo_engine.catalog()
    catalog.providers = ["openpcdet", "yolo"]
    catalog.provider_profiles = {
        "openpcdet": [profile.model_dump(mode="json") for profile in catalog.profiles],
        "yolo": [profile.model_dump(mode="json") for profile in yolo_catalog.profiles],
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
