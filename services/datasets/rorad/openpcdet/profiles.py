#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/datasets/rorad/openpcdet/profiles.py
# @IDE     : vscode

from __future__ import annotations

from pathlib import Path


KITTI_CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def build_openpcdet_profiles(openpcdet_assets_root: Path) -> list[dict]:
    cfg_root = Path(__file__).resolve().parent / "configs"
    return [
        {
            "id": "pointpillar_rorad_mid360",
            "name": "OpenPCDet PointPillar (RoRad Mid360 Experimental)",
            "architecture": "PointPillar",
            "framework": "OpenPCDet",
            "weight_name": "pointpillar_kitti.pth",
            "description": "Experimental RoRad/Mid360 inference profile using the KITTI PointPillar checkpoint.",
            "cfg_file": str(cfg_root / "pointpillar_rorad_mid360.yaml"),
            "ckpt_path": str(openpcdet_assets_root / "pointpillar_kitti.pth"),
            "class_names": KITTI_CLASS_NAMES,
            "preprocess_point_cloud_range": [-5.0, -35.0, -3.0, 61.56, 41.8, 2.0],
            "default_score_threshold": 0.1,
        }
    ]
