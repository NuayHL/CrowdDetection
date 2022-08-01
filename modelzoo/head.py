import torch
import torch.nn as nn

from modelzoo.common import conv_batch

class Yolov3_head(nn.Module):
    def __init__(self, classes, anchors_per_grid):
        super(Yolov3_head, self).__init__()
        self.p5_head = nn.Sequential(
            conv_batch(512, 1024),
            conv_batch(1024, (1 + 4 + classes) * anchors_per_grid, kernel_size=1, padding=0))
        self.p4_head = nn.Sequential(
            conv_batch(256, 512),
            conv_batch(512, (1 + 4 + classes) * anchors_per_grid, kernel_size=1, padding=0))
        self.p3_head = nn.Sequential(
            conv_batch(128, 256),
            conv_batch(256, (1 + 4 + classes) * anchors_per_grid, kernel_size=1, padding=0))
    def forward(self, p3, p4, p5):
        p3 = self.p3_head(p3)
        p4 = self.p4_head(p4)
        p5 = self.p5_head(p5)
        return p3, p4, p5
