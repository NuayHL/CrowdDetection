import torch
import torch.nn as nn

from modelzoo.common import conv_batch

"""
head init format:
    if using anchor: 
        def __init__(self, classes, anchors_per_grid, p3c)
    if not using anchor:
        def __init__(self, classes, p3c)
"""

class Yolov3_head(nn.Module):
    def __init__(self, classes, anchors_per_grid, p3_channels=128):
        super(Yolov3_head, self).__init__()
        p3c = p3_channels
        self.p5_head = nn.Sequential(
            conv_batch(p3c*4, p3c*8),
            conv_batch(p3c*8, (1 + 4 + classes) * anchors_per_grid, kernel_size=1, padding=0))
        self.p4_head = nn.Sequential(
            conv_batch(p3c*2, p3c*4),
            conv_batch(p3c*4, (1 + 4 + classes) * anchors_per_grid, kernel_size=1, padding=0))
        self.p3_head = nn.Sequential(
            conv_batch(p3c, p3c*2),
            conv_batch(p3c*2, (1 + 4 + classes) * anchors_per_grid, kernel_size=1, padding=0))
    def forward(self, p3, p4, p5):
        p3 = self.p3_head(p3)
        p4 = self.p4_head(p4)
        p5 = self.p5_head(p5)
        return p3, p4, p5

class Retina_head(nn.Module):
    def __init__(self, classes, anchors_per_grid, p3c=256):
        super(Retina_head, self).__init__()
        self.reg_branch = nn.Sequential(nn.Conv2d(p3c, p3c, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(p3c, p3c, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(p3c, p3c, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(p3c, p3c, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(p3c, anchors_per_grid * 4, kernel_size=3, padding=1))

        self.cls_branch = nn.Sequential(nn.Conv2d(p3c, p3c, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(p3c, p3c, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(p3c, p3c, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(p3c, p3c, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(p3c, anchors_per_grid * classes, kernel_size=3, padding=1),
                                        nn.Sigmoid())
    def forward(self,*feature_maps):
        cls = []
        reg = []
        for map in feature_maps:
            cls_i = self.cls_branch(map)
            reg_i = self.reg_branch(map)
            out = torch.flatten(out, start_dim=2)
            out_split = torch.split(out, int(out.shape[1] / self.num_anchors), dim=1)
            out = torch.cat(out_split, dim=2)


def build_head(name):
    if name == 'yolov3_head':
        return Yolov3_head
    else:
        raise NotImplementedError('No head named %s'%name)