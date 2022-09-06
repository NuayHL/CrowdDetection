import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from modelzoo.common import conv_nobias_bn_lrelu

class Yolov3_neck(nn.Module):
    def __init__(self, p3_channels=256):
        super(Yolov3_neck, self).__init__()
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

class Retina_neck(nn.Module):
    def __init__(self, p3c = 256):
        super(Retina_neck, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(p3c*4, p3c, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(p3c, p3c, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(p3c*2, p3c, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(p3c, p3c, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(p3c, p3c, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(p3c, p3c, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(p3c*4, p3c, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(p3c, p3c, kernel_size=3, stride=2, padding=1)

    def forward(self, C3, C4, C5):
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return P3_x, P4_x, P5_x, P6_x, P7_x

from modelzoo.common import DWConv, BaseConv, CSPLayer
class PAFPN(nn.Module):
    def __init__(self, p3c=128, depth=1.0, depthwise=False, act='silu'):
        super(PAFPN, self).__init__()
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

def build_neck(name):
    '''return neckClass, ratio on p3c'''
    if name == 'yolov3_neck':
        return Yolov3_neck, 0.5
    elif name == 'retina_neck':
        return Retina_neck, 1.0
    elif name == 'pafpn':
        return PAFPN, 1.0
    else:
        raise NotImplementedError('No neck named %s'%name)

if __name__ == "__main__":
    p3 = torch.rand((1, 256, 32, 32))
    p4 = torch.rand((1, 512, 16, 16))
    p5 = torch.rand((1, 1024, 8, 8))

    fpn = PAFPN(p3c=256)
    result = fpn(p3,p4,p5)
    for i in result:
        print(i.shape)