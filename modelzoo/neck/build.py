import torch

from utility.register import Register

NeckRegister = Register('neck')

def build_neck(name):
    """return neckClass"""
    try:
        return NeckRegister[name]
    except:
        raise NotImplementedError('No neck named %s'%name)

if __name__ == "__main__":
    from modelzoo.build_models import numofParameters
    p3 = torch.rand((1, 256, 32, 32))
    p4 = torch.rand((1, 512, 16, 16))
    p5 = torch.rand((1, 1024, 8, 8))

    neck = build_neck('pafpn')
    fpn = neck(256)
    print("Num of Parameters: %.2fM" % (numofParameters(fpn) / 1e6))
    result = fpn(p3, p4, p5)
    for i in result:
        print(i.shape)