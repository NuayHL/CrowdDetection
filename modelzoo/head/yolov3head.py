import torch
from torch import nn as nn

from modelzoo.common import conv_nobias_bn_lrelu


class Yolov3_head(nn.Module):
    def __init__(self, classes, anchors_per_grid, p3_channels=128):
        super(Yolov3_head, self).__init__()
        p3c = p3_channels
        self.classes = classes
        self.anchors = anchors_per_grid
        self.p5_head = nn.Sequential(
            conv_nobias_bn_lrelu(p3c * 4, p3c * 8),
            conv_nobias_bn_lrelu(p3c * 8, (1 + 4 + classes) * anchors_per_grid, kernel_size=1, padding=0))
        self.p4_head = nn.Sequential(
            conv_nobias_bn_lrelu(p3c * 2, p3c * 4),
            conv_nobias_bn_lrelu(p3c * 4, (1 + 4 + classes) * anchors_per_grid, kernel_size=1, padding=0))
        self.p3_head = nn.Sequential(
            conv_nobias_bn_lrelu(p3c, p3c * 2),
            conv_nobias_bn_lrelu(p3c * 2, (1 + 4 + classes) * anchors_per_grid, kernel_size=1, padding=0))
    def forward(self, *p):
        p3 = self.p3_head(p[0])
        p4 = self.p4_head(p[1])
        p5 = self.p5_head(p[2])
        return self._result_parse((p3, p4, p5))

    def _result_parse(self, triple):
        '''
        flatten the results according to the format of anchors
        '''
        out = []
        for fp in triple:
            fp = torch.flatten(fp, start_dim=2)
            split = torch.split(fp, int(fp.shape[1] / self.anchors), dim=1)
            fp = torch.cat(split, dim=2)
            out.append(fp)
        out = torch.cat(out,dim=2)
        return out
