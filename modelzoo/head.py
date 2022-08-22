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
        return cls, reg

def build_head(name):
    if name == 'yolov3_head':
        return Yolov3_head
    elif name == 'retina_head':
        return Retina_head
    else:
        raise NotImplementedError('No head named %s'%name)

if __name__ == '__main__':
    input3 = torch.ones([1,256,64,64])
    input4 = torch.ones([1,256,32,32])
    input5 = torch.ones([1,256,16,16])
    input6 = torch.ones([1,256,8,8])
    input7 = torch.ones([1,256,4,4])
    head = Retina_head(2, 4)
    cls, reg = head(input3,input4,input5,input6,input7)
    print(cls.shape,reg.shape)