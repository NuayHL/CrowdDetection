import torch
from torch import nn as nn

from modelzoo.common import DWConv, BaseConv, CSPLayer
from modelzoo.neck.build import NeckRegister

@NeckRegister.register
@NeckRegister.register('pafpn')
class PAFPN(nn.Module):
    def __init__(self, p3c=128, depth=1.0, depthwise=False, act='silu'):
        super(PAFPN, self).__init__()
        self.p3c_r = 1.0
        self.in_channels = [int(p3c), int(p3c * 2), int(p3c * 4)]
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        Conv = DWConv if depthwise else BaseConv
        width = 1.0

        self.p5_lateral_conv = BaseConv(int(self.in_channels[2] * width),int(self.in_channels[1] * width),
                                     kernel_size=1, stride=1, act=act)
        self.p45_to_p4 = CSPLayer(int(2 * self.in_channels[1] * width), int(self.in_channels[1] * width),
                                  round(3 * depth), shortcut=False, depthwise=depthwise, act=act)
        self.p4_reduce_conv = BaseConv(int(self.in_channels[1] * width),int(self.in_channels[0] * width),
                                     kernel_size=1, stride=1, act=act)
        self.p34_to_p3 = CSPLayer(int(2 * self.in_channels[0] * width), int(self.in_channels[0] * width),
                                  round(3 * depth), shortcut=False, depthwise=depthwise, act=act)
        self.p3_to_p4 = Conv(int(self.in_channels[0] * width), int(self.in_channels[0] * width),
                             kernel_size=3, stride=2, act=act)
        self.p34_to_p4 = CSPLayer(int(2 * self.in_channels[0] * width), int(self.in_channels[1] * width),
                                  round(3 * depth), shortcut=False, depthwise=depthwise, act=act)
        self.p4_to_p5 = Conv(int(self.in_channels[1] * width), int(self.in_channels[1] * width),
                             kernel_size=3, stride=2, act=act)
        self.p45_to_p5 = CSPLayer(int(2 * self.in_channels[1] * width), int(self.in_channels[2] * width),
                                  round(3 * depth), shortcut=False, depthwise=depthwise, act=act)

    def forward(self, p3, p4, p5):
        p5_1 = self.p5_lateral_conv(p5)
        p5_up = self.upsample(p5_1)
        p4 = self.p45_to_p4(torch.cat([p5_up,p4],dim=1))

        p4_1 = self.p4_reduce_conv(p4)
        p4_up = self.upsample(p4_1)
        p3_out = self.p34_to_p3(torch.cat([p4_up,p3],dim=1))

        p3_down = self.p3_to_p4(p3_out)
        p4_out = self.p34_to_p4(torch.cat([p3_down,p4_1],dim=1))

        p4_down = self.p4_to_p5(p4_out)
        p5_out = self.p45_to_p5(torch.cat([p4_down,p5_1],dim=1))

        return p3_out, p4_out, p5_out
