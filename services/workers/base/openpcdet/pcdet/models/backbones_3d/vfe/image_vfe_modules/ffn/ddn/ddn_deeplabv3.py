#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/workers/base/openpcdet/pcdet/models/backbones_3d/vfe/image_vfe_modules/ffn/ddn/ddn_deeplabv3.py
# @IDE     : vscode

# Reference: adapted from OpenPCDet.
# Source: https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_3d/vfe/image_vfe_modules/ffn/ddn/ddn_deeplabv3.py

from .ddn_template import DDNTemplate

try:
    import torchvision
except:
    pass


class DDNDeepLabV3(DDNTemplate):

    def __init__(self, backbone_name, **kwargs):
        """
        Initializes DDNDeepLabV3 model
        Args:
            backbone_name: string, ResNet Backbone Name [ResNet50/ResNet101]
        """
        if backbone_name == "ResNet50":
            constructor = torchvision.models.segmentation.deeplabv3_resnet50
        elif backbone_name == "ResNet101":
            constructor = torchvision.models.segmentation.deeplabv3_resnet101
        else:
            raise NotImplementedError

        super().__init__(constructor=constructor, **kwargs)
