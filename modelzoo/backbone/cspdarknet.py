from torch import nn as nn

from modelzoo.common import DWConv, BaseConv, Focus, CSPLayer, SPPBottleneck
from modelzoo.attention import CBAMBlock


class CSPDarknet(nn.Module):
    def __init__(self, depth=1.0, width=1.0, depthwise=False, act='silu'):
        super(CSPDarknet, self).__init__()
        base_channels = int(width * 64)
        base_depth = max(round(depth * 3), 1)
        Conv = DWConv if depthwise else BaseConv
        self.p3c = base_channels * 4

        self.stem = Focus(3, base_channels, kernel_size=3, act=act)

        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2, base_channels * 2,
                n=base_depth, depthwise=depthwise, act=act
            ),
        )
        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4, base_channels * 4,
                n=base_depth * 3, depthwise=depthwise, act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8, base_channels * 8,
                n=base_depth * 3, depthwise=depthwise, act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16, base_channels * 16, n=base_depth,
                shortcut=False, depthwise=depthwise, act=act,
            ),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        p3 = self.dark3(x)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        return p3, p4, p5

class CSPDarknet_CBAM(nn.Module):
    """Adding CBAMBlock in each dark layer"""
    def __init__(self, depth=1.0, width=1.0, depthwise=False, act='silu'):
        super(CSPDarknet_CBAM, self).__init__()
        base_channels = int(width * 64)
        base_depth = max(round(depth * 3), 1)
        Conv = DWConv if depthwise else BaseConv
        self.p3c = base_channels * 4

        self.stem = Focus(3, base_channels, kernel_size=3, act=act)

        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CBAMBlock(base_channels * 2, 7, 16),
            CSPLayer(
                base_channels * 2, base_channels * 2,
                n=base_depth, depthwise=depthwise, act=act
            ),
        )
        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CBAMBlock(base_channels * 4, 7, 16),
            CSPLayer(
                base_channels * 4, base_channels * 4,
                n=base_depth * 3, depthwise=depthwise, act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CBAMBlock(base_channels * 8, 7, 16),
            CSPLayer(
                base_channels * 8, base_channels * 8,
                n=base_depth * 3, depthwise=depthwise, act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CBAMBlock(base_channels * 16, 7, 16),
            CSPLayer(
                base_channels * 16, base_channels * 16, n=base_depth,
                shortcut=False, depthwise=depthwise, act=act,
            ),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        p3 = self.dark3(x)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        return p3, p4, p5
