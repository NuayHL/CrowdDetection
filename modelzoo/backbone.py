import torch.nn as nn

from modelzoo.common import conv_batch
from modelzoo.common import DarkResidualBlock
from modelzoo.common import make_layers

def build_backbone(name):
    '''return backboneClass, output p3c'''
    if name == 'darknet53':
        return Darknet53, 256
    else:
        raise NotImplementedError('No backbone named %s' % (name))

class Darknet53(nn.Module):
    def __init__(self, res_block=DarkResidualBlock):
        super(Darknet53, self).__init__()

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = make_layers(1, res_block, 64)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = make_layers(2, res_block, 128)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = make_layers(8, res_block, 256)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = make_layers(8, res_block, 512)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = make_layers(4, res_block, 1024)

    def forward(self,x):
        ''' p3 channel: 256 '''
        x = self.conv1(x)
        x = self.residual_block1(self.conv2(x))
        x = self.residual_block2(self.conv3(x))
        p3 = self.residual_block3(self.conv4(x))
        p4 = self.residual_block4(self.conv5(p3))
        p5 = self.residual_block5(self.conv6(p4))
        return p3, p4, p5



