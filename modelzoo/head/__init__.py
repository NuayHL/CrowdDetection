from utility.register import load_modules
load_modules('modelzoo.head', __file__)

# Import for the old ckpt
from modelzoo.head.yoloxhead import YOLOX_head

from .build import build_head
__all__ = ["build_head"]