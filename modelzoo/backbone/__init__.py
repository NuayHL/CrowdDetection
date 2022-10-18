from utility.register import load_modules
load_modules('modelzoo.backbone', __file__)

from .build import build_backbone
__all__ = ["build_backbone"]
