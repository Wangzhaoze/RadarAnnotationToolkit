# Reference: adapted from OpenPCDet.
# Source: https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_3d/vfe/image_vfe_modules/ffn/__init__.py

from .depth_ffn import DepthFFN

__all__ = {
    'DepthFFN': DepthFFN
}
