from utility.register import load_modules
load_modules('utility.assign', __file__)

from .build import get_assign_method
__all__ = ["get_assign_method"]