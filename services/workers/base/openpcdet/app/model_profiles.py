from __future__ import annotations

import os
from pathlib import Path


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
    "pointpillar_rorad_mid360": {
        "id": "pointpillar_rorad_mid360",
        "name": "OpenPCDet PointPillar (RoRad Mid360 Experimental)",
        "architecture": "PointPillar",
        "framework": "OpenPCDet",
        "weight_name": "pointpillar_kitti.pth",
        "description": "Experimental RoRad/Mid360 inference profile using the KITTI PointPillar checkpoint.",
        "cfg_file": str(KITTI_CFG_ROOT / "pointpillar_rorad_mid360.yaml"),
        "ckpt_path": str(OPENPCDET_ASSETS_ROOT / "pointpillar_kitti.pth"),
        "class_names": KITTI_CLASS_NAMES,
        "preprocess_point_cloud_range": [-5.0, -35.0, -3.0, 61.56, 41.8, 2.0],
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


DEFAULT_MODEL_PROFILE_ID = os.getenv("DEFAULT_MODEL_PROFILE_ID", "pv_rcnn_kitti")
