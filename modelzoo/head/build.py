import torch

from modelzoo.head.retinahead import Retina_head
from modelzoo.head.yolov3head import Yolov3_head
from modelzoo.head.yoloxhead import YOLOX_head

"""
head init format:
    if using anchor: 
        def __init__(self, classes, anchors_per_grid, p3c)
    if not using anchor:
        def __init__(self, classes, p3c)
"""

def build_head(name):
    if name == 'yolov3_head':
        return Yolov3_head
    elif name == 'retina_head':
        return Retina_head
    elif name == 'yolox_head':
        return YOLOX_head
    else:
        raise NotImplementedError('No head named %s'%name)

if __name__ == '__main__':
    input3 = torch.ones([1,128,64,64])
    input4 = torch.ones([1,256,32,32])
    input5 = torch.ones([1,512,16,16])
    # input6 = torch.ones([1,256,8,8])
    # input7 = torch.ones([1,256,4,4])
    head = YOLOX_head(2, 2, 128)
    obj = head(input3,input4,input5)
    print(obj.shape)