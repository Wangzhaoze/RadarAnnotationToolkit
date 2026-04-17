#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/workers/base/openpcdet/pcdet/models/backbones_2d/__init__.py
# @IDE     : vscode

# Reference: adapted from OpenPCDet.
# Source: https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_2d/__init__.py

from .base_bev_backbone import BaseBEVBackbone, BaseBEVBackboneV1

__all__ = {
    'BaseBEVBackbone': BaseBEVBackbone,
    'BaseBEVBackboneV1': BaseBEVBackboneV1
}
