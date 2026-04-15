# Reference: adapted from OpenPCDet.
# Source: https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_2d/__init__.py

from .base_bev_backbone import BaseBEVBackbone, BaseBEVBackboneV1

__all__ = {
    'BaseBEVBackbone': BaseBEVBackbone,
    'BaseBEVBackboneV1': BaseBEVBackboneV1
}
