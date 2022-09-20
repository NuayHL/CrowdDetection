from .build import build_head

from modelzoo.head.retinahead import Retina_head
from modelzoo.head.yolov3head import Yolov3_head
from modelzoo.head.yoloxhead import YOLOX_head

__all__ = ["build_head",
           "Retina_head",
           "YOLOX_head",
           "Yolov3_head"]