# Reference: adapted from OpenPCDet.
# Source: https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_3d/vfe/image_vfe_modules/ffn/ddn/__init__.py

from .ddn_deeplabv3 import DDNDeepLabV3

__all__ = {
    'DDNDeepLabV3': DDNDeepLabV3
}
