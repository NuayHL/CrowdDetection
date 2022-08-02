import torch
import torch.nn as nn

from modelzoo.backbone import Darknet53
from modelzoo.neck import Yolov3_neck
from modelzoo.head import Yolov3_head

class Yolov3(nn.Module):
    def __init__(self, classes, anchors_per_grid):
        super(Yolov3, self).__init__()
        self.backbone = Darknet53()
        self.neck = Yolov3_neck()
        self.head = Yolov3_head(classes, anchors_per_grid)

    def forward(self, sample):
        if self.training:
            return self.training_loss(sample)
        else:
            return self.inferencing(sample)
    def training_loss(self,sample):
        pass
    def inferencing(self, sample):
        pass
    def core(self,sample):
        p3, p4, p5 = self.backbone(sample['imgs'])
        p3, p4, p5 = self.neck(p3, p4, p5)
        p3, p4, p5 = self.head(p3, p4, p5)
        return p3, p4, p5

if __name__ == '__main__':
    model = Yolov3(1, 4).cuda()
    input = torch.rand((1,3,640,640)).cuda()
    input = {'imgs':input}
    output = model.core(input)
    print(output[0].shape,output[1].shape,output[2].shape)
