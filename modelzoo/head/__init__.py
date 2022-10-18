from utility.register import load_modules
load_modules('modelzoo.head', __file__)

from .build import build_head
__all__ = ["build_head"]