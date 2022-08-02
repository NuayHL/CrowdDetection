import torch
import torch.nn as nn

from utility.assign import AnchorAssign
from utility.anchors import generateAnchors

class Yolov3(nn.Module):
    def __init__(self, config, backbone, neck, head):
        super(Yolov3, self).__init__()
        self.config = config
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, sample):
        if self.training:
            return self.training_loss(sample)
        else:
            return self.inferencing(sample)
    def core(self,sample):
        p3, p4, p5 = self.backbone(sample['imgs'])
        p3, p4, p5 = self.neck(p3, p4, p5)
        p3, p4, p5 = self.head(p3, p4, p5)
        return p3, p4, p5
    def training_loss(self,sample):
        pass
    def inferencing(self, sample):
        pass


