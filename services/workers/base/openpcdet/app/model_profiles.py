#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/workers/base/openpcdet/app/model_profiles.py
# @IDE     : vscode

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType


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


MODEL_PROFILES = {
    "pv_rcnn_kitti": {
        "id": "pv_rcnn_kitti",
        "name": "OpenPCDet PV-RCNN (KITTI)",
        "architecture": "PV-RCNN",
        "framework": "OpenPCDet",
        "weight_name": "pv_rcnn_kitti.pth",
        "description": "PV-RCNN checkpoint pretrained on KITTI.",
        "cfg_file": str(KITTI_CFG_ROOT / "pv_rcnn.yaml"),
        "ckpt_path": str(OPENPCDET_ASSETS_ROOT / "pv_rcnn_kitti.pth"),
        "class_names": KITTI_CLASS_NAMES,
        "preprocess_point_cloud_range": [-50.0, -50.0, -5.0, 100.0, 50.0, 3.0],
        "default_score_threshold": 0.1,
    },
    "pointpillar_kitti": {
        "id": "pointpillar_kitti",
        "name": "OpenPCDet PointPillar (KITTI)",
        "architecture": "PointPillar",
        "framework": "OpenPCDet",
        "weight_name": "pointpillar_kitti.pth",
        "description": "PointPillar checkpoint pretrained on KITTI.",
        "cfg_file": str(KITTI_CFG_ROOT / "pointpillar.yaml"),
        "ckpt_path": str(OPENPCDET_ASSETS_ROOT / "pointpillar_kitti.pth"),
        "class_names": KITTI_CLASS_NAMES,
        "preprocess_point_cloud_range": [-50.0, -50.0, -5.0, 100.0, 50.0, 3.0],
        "default_score_threshold": 0.1,
    },
    "second_kitti": {
        "id": "second_kitti",
        "name": "OpenPCDet SECOND (KITTI)",
        "architecture": "SECOND",
        "framework": "OpenPCDet",
        "weight_name": "second_kitti.pth",
        "description": "SECOND checkpoint pretrained on KITTI.",
        "cfg_file": str(KITTI_CFG_ROOT / "second.yaml"),
        "ckpt_path": str(OPENPCDET_ASSETS_ROOT / "second_kitti.pth"),
        "class_names": KITTI_CLASS_NAMES,
        "preprocess_point_cloud_range": [-50.0, -50.0, -5.0, 100.0, 50.0, 3.0],
        "default_score_threshold": 0.1,
    },
    "pointpillar_multihead_nuscenes": {
        "id": "pointpillar_multihead_nuscenes",
        "name": "OpenPCDet PointPillar Multi-Head (nuScenes)",
        "architecture": "PointPillar",
        "framework": "OpenPCDet",
        "weight_name": "pointpillar_multihead_nuscenes.pth",
        "description": "Multi-head PointPillar checkpoint pretrained on nuScenes.",
        "cfg_file": str(NUSCENES_CFG_ROOT / "cbgs_pp_multihead.yaml"),
        "ckpt_path": str(OPENPCDET_ASSETS_ROOT / "pointpillar_multihead_nuscenes.pth"),
        "class_names": NUSCENES_CLASS_NAMES,
        "preprocess_point_cloud_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        "default_score_threshold": 0.1,
    },
    "centerpoint_nuscenes": {
        "id": "centerpoint_nuscenes",
        "name": "OpenPCDet CenterPoint (nuScenes)",
        "architecture": "CenterPoint",
        "framework": "OpenPCDet",
        "weight_name": "centerpoint_nuscenes.pth",
        "description": "CenterPoint dynamic-pillar checkpoint pretrained on nuScenes.",
        "cfg_file": str(NUSCENES_CFG_ROOT / "cbgs_dyn_pp_centerpoint.yaml"),
        "ckpt_path": str(OPENPCDET_ASSETS_ROOT / "centerpoint_nuscenes.pth"),
        "class_names": NUSCENES_CLASS_NAMES,
        "preprocess_point_cloud_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        "default_score_threshold": 0.1,
    },
}


def _load_profile_module(dataset_id: str, profiles_path: Path) -> ModuleType:
    module_name = f"radar_annotation_openpcdet_profiles_{dataset_id}"
    spec = importlib.util.spec_from_file_location(module_name, profiles_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load OpenPCDet profile plugin: {profiles_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _discover_dataset_profiles() -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    if not DATASET_PLUGIN_ROOT.exists():
        return profiles
    for dataset_dir in sorted(DATASET_PLUGIN_ROOT.iterdir()):
        profiles_path = dataset_dir / "openpcdet" / "profiles.py"
        if not profiles_path.exists():
            continue
        module = _load_profile_module(dataset_dir.name, profiles_path)
        build_profiles = getattr(module, "build_openpcdet_profiles", None)
        if not callable(build_profiles):
            continue
        for profile in build_profiles(OPENPCDET_ASSETS_ROOT) or []:
            profile_id = str(profile.get("id") or "").strip()
            if not profile_id:
                continue
            if profile_id in MODEL_PROFILES or profile_id in profiles:
                raise ValueError(f"Duplicate OpenPCDet profile id: {profile_id}")
            profiles[profile_id] = profile
    return profiles


MODEL_PROFILES.update(_discover_dataset_profiles())


DEFAULT_MODEL_PROFILE_ID = os.getenv("DEFAULT_MODEL_PROFILE_ID", "pv_rcnn_kitti")
