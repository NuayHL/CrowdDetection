import torch
from modelzoo.neck.pafpn import PAFPN
from modelzoo.neck.retinafpn import Retina_neck
from modelzoo.neck.yolov3neck import Yolov3_neck
from modelzoo.neck.abneck import ABNeck
def build_neck(name):
    '''return neckClass, ratio on p3c'''
    if name == 'yolov3_neck':
        return Yolov3_neck, 0.5
    elif name == 'retina_neck':
        return Retina_neck, 1.0
    elif name == 'pafpn':
        return PAFPN, 1.0
    elif name == 'abneck':
        return ABNeck, 1.0

    else:
        raise NotImplementedError('No neck named %s'%name)

if __name__ == "__main__":
    from modelzoo.build_models import numofParameters
    p3 = torch.rand((1, 256, 32, 32))
    p4 = torch.rand((1, 512, 16, 16))
    p5 = torch.rand((1, 1024, 8, 8))

    fpn = PAFPN(256)
    print("Num of Parameters: %.2fM" % (numofParameters(fpn) / 1e6))
    result = fpn(p3, p4, p5)
    for i in result:
        print(i.shape)