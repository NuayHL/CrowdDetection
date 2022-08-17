import torch
import torch.nn as nn


class RetinaNet(nn.Module):
    def __init__(self, config, backbone, neck, head):
        super(RetinaNet, self).__init__()
        self.config = config
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, sample):
        if self.training:
            return self.training_loss(sample)
        else:
            return self.inferencing(sample)

    def core(self,input):
        p3, p4, p5 = self.backbone(input)
        p3, p4, p5, p6, p7 = self.neck(p3,p4,p5)
        ######## here #########

    def training_loss(self): pass
    def inferencing(self): pass
