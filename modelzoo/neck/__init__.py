from utility.register import load_modules
load_modules('modelzoo.neck', __file__)

from .build import build_neck
__all__ = ["build_neck"]