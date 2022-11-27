from utility.register import load_modules
load_modules('modelzoo.backbone', __file__)

# Import for the old ckpt
from modelzoo.backbone.cspdarknet import CSPDarknet
from modelzoo.backbone.darknet import Darknet53

from .build import build_backbone
__all__ = ["build_backbone"]
