import torch
from torch import nn as nn


class Retina_head(nn.Module):
    def __init__(self, classes, anchors_per_grid, p3c=256):
        super(Retina_head, self).__init__()
        self.num_anchors = anchors_per_grid
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
                                        nn.Conv2d(p3c, anchors_per_grid * (1 + classes), kernel_size=3, padding=1))

    def forward(self,*feature_maps):
        cls = []
        reg = []
        for map in feature_maps:
            cls_i = self.cls_branch(map)
            reg_i = self.reg_branch(map)
            cls_i = torch.flatten(cls_i, start_dim=2)
            reg_i = torch.flatten(reg_i, start_dim=2)
            cls_i_split = torch.split(cls_i, int(cls_i.shape[1] / self.num_anchors), dim=1)
            reg_i_split = torch.split(reg_i, int(reg_i.shape[1] / self.num_anchors), dim=1)
            cls_i = torch.cat(cls_i_split, dim=2)
            reg_i = torch.cat(reg_i_split, dim=2)
            cls.append(cls_i)
            reg.append(reg_i)
        cls = torch.cat(cls, dim=2)
        reg = torch.cat(reg, dim=2)
        return torch.cat([reg, cls], dim=1)
