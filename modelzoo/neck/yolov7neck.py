import torch
from torch import nn as nn

from modelzoo.neck.build import NeckRegister
from modelzoo.common import YOLOv7_common

Conv = YOLOv7_common.Conv
Multi_Concat_Block = YOLOv7_common.Multi_Concat_Block
Transition_Block = YOLOv7_common.Transition_Block

@NeckRegister.register
@NeckRegister.register('yolov7_neck')
class Yolov7_neck(nn.Module):
    def __init__(self, p3_channels=512):
        super(Yolov7_neck, self).__init__()
        self.p3c_r = 0.25
        ids = [-1, -3, -5, -6]
        transition_channels = 32
        panet_channels = 32
        n = 4
        e = 2
        assert p3_channels == 512
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.sppcspc = SPPCSPC(transition_channels * 32, transition_channels * 16)
        self.conv_for_P5 = Conv(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2 = Conv(transition_channels * 32, transition_channels * 8)
        self.conv3_for_upsample1 = Multi_Concat_Block(transition_channels * 16, panet_channels * 4,
                                                      transition_channels * 8, e=e, n=n, ids=ids)

        self.conv_for_P4 = Conv(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1 = Conv(transition_channels * 16, transition_channels * 4)
        self.conv3_for_upsample2 = Multi_Concat_Block(transition_channels * 8, panet_channels * 2,
                                                      transition_channels * 4, e=e, n=n, ids=ids)

        self.down_sample1 = Transition_Block(transition_channels * 4, transition_channels * 4)
        self.conv3_for_downsample1 = Multi_Concat_Block(transition_channels * 16, panet_channels * 4,
                                                        transition_channels * 8, e=e, n=n, ids=ids)

        self.down_sample2 = Transition_Block(transition_channels * 8, transition_channels * 8)
        self.conv3_for_downsample2 = Multi_Concat_Block(transition_channels * 32, panet_channels * 8,
                                                        transition_channels * 16, e=e, n=n, ids=ids)

    def forward(self,p3, p4, p5):
        P5 = self.sppcspc(p5)
        P5_conv = self.conv_for_P5(P5)
        P5_upsample = self.upsample(P5_conv)

        P4 = torch.cat([self.conv_for_feat2(p4), P5_upsample], 1)
        P4 = self.conv3_for_upsample1(P4)
        P4_conv = self.conv_for_P4(P4)
        P4_upsample = self.upsample(P4_conv)

        P3 = torch.cat([self.conv_for_feat1(p3), P4_upsample], 1)
        P3 = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)
        # P3 80, 80, 128
        # P4 40, 40, 256
        # P5 20, 20, 512
        return P3, P4, P5

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))
