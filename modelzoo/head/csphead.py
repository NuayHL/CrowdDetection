import torch.nn as nn

class CenterScalehead(nn.Module):
    def __init__(self, classes, anchors_per_grid, p3c, depth=1.0, act='silu'):
        super(CenterScalehead, self).__init__()


    def forward(self, *levels):
        pass