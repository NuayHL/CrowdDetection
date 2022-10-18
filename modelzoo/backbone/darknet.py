from torch import nn as nn

from modelzoo.common import DarkResidualBlock, conv_nobias_bn_lrelu, make_layers
from modelzoo.backbone.build import BackboneRegister

@BackboneRegister.register
@BackboneRegister.register('darknet53')
class Darknet53(nn.Module):
    def __init__(self, res_block=DarkResidualBlock):
        super(Darknet53, self).__init__()
        Darknet53.p3c = 256
        self.conv1 = conv_nobias_bn_lrelu(3, 32)
        self.conv2 = conv_nobias_bn_lrelu(32, 64, stride=2)
        self.residual_block1 = make_layers(1, res_block, 64)
        self.conv3 = conv_nobias_bn_lrelu(64, 128, stride=2)
        self.residual_block2 = make_layers(2, res_block, 128)
        self.conv4 = conv_nobias_bn_lrelu(128, 256, stride=2)
        self.residual_block3 = make_layers(8, res_block, 256)
        self.conv5 = conv_nobias_bn_lrelu(256, 512, stride=2)
        self.residual_block4 = make_layers(8, res_block, 512)
        self.conv6 = conv_nobias_bn_lrelu(512, 1024, stride=2)
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
