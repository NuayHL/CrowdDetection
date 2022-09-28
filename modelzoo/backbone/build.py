import torch

from modelzoo.backbone.cspdarknet import CSPDarknet
from modelzoo.backbone.darknet import Darknet53
from modelzoo.backbone.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


def build_backbone(name):
    '''return backboneClass, output p3c'''
    if name == 'darknet53':
        return Darknet53, 256
    elif name == 'resnet18':
        return resnet18, 128
    elif name == 'resnet34':
        return resnet34, 128
    elif name == 'resnet50':
        return resnet50, 512
    elif name == 'resnet101':
        return resnet101, 512
    elif name == 'resnet152':
        return resnet152, 512
    elif name == 'cspdarknet':
        return CSPDarknet, None
    else:
        raise NotImplementedError('No backbone named %s' % (name))


if __name__ == "__main__":
    dummy_input = torch.ones([1,3,64,64])
    model = CSPDarknet()
    output = model(dummy_input)
    print(output[0].shape,output[1].shape,output[2].shape)


