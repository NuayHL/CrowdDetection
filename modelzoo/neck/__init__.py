from utility.register import load_modules
load_modules('modelzoo.neck', __file__)

# Import for the old ckpt
from modelzoo.neck.pafpn import PAFPN

from .build import build_neck
__all__ = ["build_neck"]