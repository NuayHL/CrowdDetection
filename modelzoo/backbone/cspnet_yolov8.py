from torch import nn as nn

from modelzoo.backbone.build import BackboneRegister
from modelzoo.common import YOLOv8_common

Conv = YOLOv8_common.Conv
C2f = YOLOv8_common.C2f
SPPF = YOLOv8_common.SPPF

# the structure of the YOLOv8 is referring to the official code and third party code
@BackboneRegister.register
@BackboneRegister.register('cspnet-YOLOv8')
class Cspnet_YOLOv8(nn.Module):
    def __init__(self, width=1.0, depth=1.0, ratio=1.0):
        super().__init__()
        base_channels = int(64 * width)
        base_depth = max(round(3 * depth), 1)
        self.p3c = base_channels * 4

        self.stem = Conv(3, base_channels, 3, 2)

        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * ratio), 3, 2),
            C2f(int(base_channels * 16 * ratio), int(base_channels * 16 * ratio), base_depth, True),
            SPPF(int(base_channels * 16 * ratio), int(base_channels * 16 * ratio), k=5)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        p3 = self.dark3(x) # 256 80 80
        p4 = self.dark4(p3) # 512 40 40
        p5 = self.dark5(p4) # 1024 20 20
        return p3, p4, p5


