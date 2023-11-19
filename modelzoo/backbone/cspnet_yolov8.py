import torch
from torch import nn as nn

from modelzoo.backbone.build import BackboneRegister
from modelzoo.common import YOLOv8_common

Conv = YOLOv8_common.Conv
C2f = YOLOv8_common.C2f
SPPF = YOLOv8_common.SPPF

# the structure of the YOLOv7 is referring to the official code and third party code
# https://github.com/WongKinYiu/yolov7
# https://github.com/bubbliiiing/yolov7-pytorch/tree/master/nets
@BackboneRegister.register
@BackboneRegister.register('cspnet-YOLOv8')
class Backbone(nn.Module):
    def __init__(self, width=1.0, base_depth, deep_mul):
        super().__init__()
        base_channels = int(64 * width)
        base_depth = max(round(d))
        self.p3c = base_channels * 4

        # -----------------------------------------------#
        #   输入图片是3, 640, 640
        # -----------------------------------------------#
        # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)

        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )
        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        # -----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        # -----------------------------------------------#
        p3 = self.dark3(x)
        # -----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        # -----------------------------------------------#
        p4 = self.dark4(p3)
        # -----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        # -----------------------------------------------#
        p5 = self.dark5(p4)
        return p3, p4, p5


