import torch.nn as nn
from modelzoo.common import RepVGGBlock

class EfficientRep(nn.Module):
    def __init__(self, channels_list, block = RepVGGBlock):
        super(EfficientRep, self).__init__()

        self.stem = block(in_channels=3, out_channels=0)


    def forward(self, input):
        pass