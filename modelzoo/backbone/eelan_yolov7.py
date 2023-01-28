import torch
from torch import nn as nn

from modelzoo.backbone.build import BackboneRegister
from modelzoo.common import YOLOv7_common

Conv = YOLOv7_common.Conv
Multi_Concat_Block = YOLOv7_common.Multi_Concat_Block
Transition_Block = YOLOv7_common.Transition_Block

# the structure of the YOLOv7 is referring to the official code and third party code
# https://github.com/WongKinYiu/yolov7
# https://github.com/bubbliiiing/yolov7-pytorch/tree/master/nets
@BackboneRegister.register
@BackboneRegister.register('E-ELAN-CSP-YOLOv7')
class Eelan_YOLOv7(nn.Module):
    def __init__(self):
        super().__init__()
        ids = [-1, -3, -5, -6]
        transition_channels = 32
        block_channels = 32
        n = 4
        self.p3c = 512
        self.stem = nn.Sequential(
            Conv(3, transition_channels, 3, 1),
            Conv(transition_channels, transition_channels * 2, 3, 2),
            Conv(transition_channels * 2, transition_channels * 2, 3, 1),
        )
        self.dark2 = nn.Sequential(
            Conv(transition_channels * 2, transition_channels * 4, 3, 2),
            Multi_Concat_Block(transition_channels * 4, block_channels * 2, transition_channels * 8, n=n, ids=ids),
        )
        self.dark3 = nn.Sequential(
            Transition_Block(transition_channels * 8, transition_channels * 4),
            Multi_Concat_Block(transition_channels * 8, block_channels * 4, transition_channels * 16, n=n, ids=ids),
        )
        self.dark4 = nn.Sequential(
            Transition_Block(transition_channels * 16, transition_channels * 8),
            Multi_Concat_Block(transition_channels * 16, block_channels * 8, transition_channels * 32, n=n, ids=ids),
        )
        self.dark5 = nn.Sequential(
            Transition_Block(transition_channels * 32, transition_channels * 16),
            Multi_Concat_Block(transition_channels * 32, block_channels * 8, transition_channels * 32, n=n, ids=ids),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        p3 = self.dark3(x)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        return p3, p4, p5


