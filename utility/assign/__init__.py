from utility.register import load_modules
load_modules('utility.assign', __file__)

# Import for the old cpkt
from utility.assign.simota import SimOTA
from utility.assign.defaultassign import AnchorAssign

from .build import get_assign_method
__all__ = ["get_assign_method"]