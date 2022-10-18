import torch
from torch import nn as nn
from torch.nn.functional import interpolate

from modelzoo.common import conv_nobias_bn_lrelu
from modelzoo.neck.build import NeckRegister

@NeckRegister.register
@NeckRegister.register('yolov3_neck')
class Yolov3_neck(nn.Module):
    def __init__(self, p3_channels=256):
        super(Yolov3_neck, self).__init__()
        self.p3c_r = 0.5
        p3c = p3_channels
        self.p5_out = self.yolo_block(p3c*2)
        self.p4_out = self.yolo_block(p3c, p3c*3)
        self.p3_out = self.yolo_block(int(p3c/2), int(p3c*3/2))
        self.p5_to_p4 = conv_nobias_bn_lrelu(p3c * 2, p3c, kernel_size=1, padding=0)
        self.p4_to_p3 = conv_nobias_bn_lrelu(p3c, int(p3c / 2), kernel_size=1, padding=0)

    def yolo_block(self, channel, ic = None):
        if not ic:
            ic = channel * 2
        return nn.Sequential(
            conv_nobias_bn_lrelu(ic, channel, kernel_size=1, padding=0),
            conv_nobias_bn_lrelu(channel, channel * 2),
            conv_nobias_bn_lrelu(channel * 2, channel, kernel_size=1, padding=0),
            conv_nobias_bn_lrelu(channel, channel * 2),
            conv_nobias_bn_lrelu(channel * 2, channel, kernel_size=1, padding=0))

    def forward(self,p3, p4, p5):
        '''p3 out channel: ic/2'''
        p5 = self.p5_out(p5)
        p5_to_p4 = interpolate(self.p5_to_p4(p5), size=(p4.shape[2], p4.shape[3]))
        p4 = self.p4_out(torch.cat((p4,p5_to_p4), dim=1))
        p4_to_p3 = interpolate(self.p4_to_p3(p4), size=(p3.shape[2], p3.shape[3]))
        p3 = self.p3_out(torch.cat((p3, p4_to_p3), dim=1))
        return p3, p4, p5
