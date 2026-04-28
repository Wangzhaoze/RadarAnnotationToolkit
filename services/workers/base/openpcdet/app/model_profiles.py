#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/workers/base/openpcdet/app/model_profiles.py
# @IDE     : vscode

from __future__ import annotations

import os
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - OpenPCDet images normally include PyYAML.
    yaml = None


OPENPCDET_SOURCE_ROOT = Path(os.getenv("OPENPCDET_SOURCE_ROOT", "/workspace/services/workers/base/openpcdet"))
OPENPCDET_ASSETS_ROOT = Path(os.getenv("OPENPCDET_ASSETS_ROOT", str(OPENPCDET_SOURCE_ROOT / "assets")))

KITTI_CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]
NUSCENES_CLASS_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

KITTI_CFG_ROOT = OPENPCDET_SOURCE_ROOT / "tools" / "cfgs" / "kitti_models"
NUSCENES_CFG_ROOT = OPENPCDET_SOURCE_ROOT / "tools" / "cfgs" / "nuscenes_models"
DEFAULT_DATASET_PLUGIN_ROOT = OPENPCDET_SOURCE_ROOT.parents[2] / "datasets"
DATASET_PLUGIN_ROOT = Path(os.getenv("DATASET_PLUGIN_ROOT", str(DEFAULT_DATASET_PLUGIN_ROOT)))
DATASET_CFG_DIR_NAME = "openpcdet_cfgs"

BASE_PROFILE_SPECS = {
    "pv_rcnn_kitti": {
        "architecture": "PV-RCNN",
        "training_dataset": "KITTI",
        "weight_name": "pv_rcnn_kitti.pth",
        "description": "PV-RCNN checkpoint pretrained on KITTI.",
        "class_names": KITTI_CLASS_NAMES,
        "preprocess_point_cloud_range": [-50.0, -50.0, -5.0, 100.0, 50.0, 3.0],
    },
    "pointpillar_kitti": {
        "architecture": "PointPillar",
        "training_dataset": "KITTI",
        "weight_name": "pointpillar_kitti.pth",
        "description": "PointPillar checkpoint pretrained on KITTI.",
        "class_names": KITTI_CLASS_NAMES,
        "preprocess_point_cloud_range": [-50.0, -50.0, -5.0, 100.0, 50.0, 3.0],
    },
    "second_kitti": {
        "architecture": "SECOND",
        "training_dataset": "KITTI",
        "weight_name": "second_kitti.pth",
        "description": "SECOND checkpoint pretrained on KITTI.",
        "class_names": KITTI_CLASS_NAMES,
        "preprocess_point_cloud_range": [-50.0, -50.0, -5.0, 100.0, 50.0, 3.0],
    },
    "pointpillar_nuscenes": {
        "architecture": "PointPillar",
        "training_dataset": "nuScenes",
        "weight_name": "pointpillar_multihead_nuscenes.pth",
        "description": "Multi-head PointPillar checkpoint pretrained on nuScenes.",
        "class_names": NUSCENES_CLASS_NAMES,
        "preprocess_point_cloud_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    },
    "centerpoint_nuscenes": {
        "architecture": "CenterPoint",
        "training_dataset": "nuScenes",
        "weight_name": "centerpoint_nuscenes.pth",
        "description": "CenterPoint dynamic-pillar checkpoint pretrained on nuScenes.",
        "class_names": NUSCENES_CLASS_NAMES,
        "preprocess_point_cloud_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    },
}


def _read_yaml(path: Path) -> dict:
    if yaml is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _spec_key_for_config(config_path: Path) -> str:
    stem = config_path.stem.lower()
    if "centerpoint" in stem:
        return "centerpoint_nuscenes"
    if "nuscenes" in stem or "multihead" in stem:
        return "pointpillar_nuscenes"
    if "pv_rcnn" in stem or "pvrcnn" in stem:
        return "pv_rcnn_kitti"
    if stem.startswith("second") or "_second" in stem:
        return "second_kitti"
    return "pointpillar_kitti"


def _extract_model_name(config: dict) -> str | None:
    model = config.get("MODEL")
    if isinstance(model, dict):
        name = model.get("NAME")
        if name:
            return str(name)
    return None


def _normalize_architecture(name: str) -> str:
    normalized = str(name or "").strip()
    compact = normalized.replace("_", "").replace("-", "").lower()
    if compact == "pvrcnn":
        return "PV-RCNN"
    if compact == "pointpillar":
        return "PointPillar"
    if compact == "centerpoint":
        return "CenterPoint"
    if compact.startswith("second"):
        return "SECOND"
    return normalized


def _extract_class_names(config: dict, fallback: list[str]) -> list[str]:
    names = config.get("CLASS_NAMES")
    if isinstance(names, list):
        clean_names = [str(name) for name in names if str(name).strip()]
        if clean_names:
            return clean_names
    return list(fallback)


def _extract_point_cloud_range(config: dict, fallback: list[float]) -> list[float]:
    data_config = config.get("DATA_CONFIG")
    if isinstance(data_config, dict):
        values = data_config.get("POINT_CLOUD_RANGE")
        if isinstance(values, list) and len(values) == 6:
            try:
                return [float(value) for value in values]
            except (TypeError, ValueError):
                pass
    return list(fallback)


def _discover_dataset_profiles() -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    if not DATASET_PLUGIN_ROOT.exists():
        return profiles
    for dataset_dir in sorted(DATASET_PLUGIN_ROOT.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("_"):
            continue
        config_dir = dataset_dir / DATASET_CFG_DIR_NAME
        if not config_dir.exists():
            continue
        for config_path in sorted(config_dir.glob("*.yaml")):
            spec = BASE_PROFILE_SPECS[_spec_key_for_config(config_path)]
            config = _read_yaml(config_path)
            architecture = _normalize_architecture(_extract_model_name(config) or spec["architecture"])
            training_dataset = spec["training_dataset"]
            config_label = config_path.stem
            network_label = f"{architecture} ({training_dataset})"
            profile_id = f"openpcdet_{dataset_dir.name}_{config_path.stem}"
            if not profile_id:
                continue
            if profile_id in profiles:
                raise ValueError(f"Duplicate OpenPCDet profile id: {profile_id}")
            profiles[profile_id] = {
                "id": profile_id,
                "name": f"OpenPCDet {network_label} / {config_label}",
                "architecture": architecture,
                "framework": "OpenPCDet",
                "weight_name": spec["weight_name"],
                "description": spec["description"],
                "cfg_file": str(config_path),
                "ckpt_path": str(OPENPCDET_ASSETS_ROOT / spec["weight_name"]),
                "class_names": _extract_class_names(config, spec["class_names"]),
                "preprocess_point_cloud_range": _extract_point_cloud_range(
                    config,
                    spec["preprocess_point_cloud_range"],
                ),
                "default_score_threshold": 0.1,
                "provider": "openpcdet",
                "task": "3d_lidar_bbox",
                "dataset_id": dataset_dir.name,
                "training_dataset": training_dataset,
                "network_label": network_label,
                "config_label": config_label,
            }
    return profiles


MODEL_PROFILES = _discover_dataset_profiles()


_configured_default_profile_id = os.getenv("DEFAULT_MODEL_PROFILE_ID", "")
_fallback_default_profile_id = next(
    (profile_id for profile_id in MODEL_PROFILES if profile_id.endswith("_pv_rcnn_kitti")),
    next(iter(MODEL_PROFILES), ""),
)
DEFAULT_MODEL_PROFILE_ID = (
    _configured_default_profile_id
    if _configured_default_profile_id in MODEL_PROFILES
    else _fallback_default_profile_id
)
