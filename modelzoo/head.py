import torch
import torch.nn as nn

from modelzoo.common import conv_nobias_bn_lrelu, BaseConv, DWConv

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
            conv_nobias_bn_lrelu(p3c * 4, p3c * 8),
            conv_nobias_bn_lrelu(p3c * 8, (1 + 4 + classes) * anchors_per_grid, kernel_size=1, padding=0))
        self.p4_head = nn.Sequential(
            conv_nobias_bn_lrelu(p3c * 2, p3c * 4),
            conv_nobias_bn_lrelu(p3c * 4, (1 + 4 + classes) * anchors_per_grid, kernel_size=1, padding=0))
        self.p3_head = nn.Sequential(
            conv_nobias_bn_lrelu(p3c, p3c * 2),
            conv_nobias_bn_lrelu(p3c * 2, (1 + 4 + classes) * anchors_per_grid, kernel_size=1, padding=0))
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

class YOLOX_head(nn.Module):
    def __init__(self, classes, anchors_per_grid, p3_channels=256, depthwise=False, width=1.0, act='silu'):
        super(YOLOX_head, self).__init__()
        self.depthwise = depthwise
        self.width = width
        self.act = act
        self.classes = classes
        self.anchors = anchors_per_grid
        self.p3c = p3_channels
        self.in_channels = [int(self.p3c), int(self.p3c * 2), int(self.p3c * 4)]

        self.stems = nn.ModuleList()
        self.cls_conv = nn.ModuleList()
        self.reg_conv = nn.ModuleList()
        self.cls_pred = nn.ModuleList()
        self.reg_pred = nn.ModuleList()
        self.obj_pred = nn.ModuleList()

        Conv = DWConv if self.depthwise else BaseConv

        for chs in self.in_channels:
            self.stems.append(Conv(in_channels=int(chs * width),
                                   out_channels=int(256 * width),
                                   kernel_size=1,stride=1, act=self.act))
            self.cls_conv.append(nn.Sequential(
                                *[Conv(in_channels=int(256 * width),
                                       out_channels=int(256 * width),
                                       kernel_size=3, stride=1, act=act),
                                  Conv(in_channels=int(256 * width),
                                       out_channels=int(256 * width),
                                       kernel_size=3, stride=1,act=act)]))
            self.reg_conv.append(nn.Sequential(
                                *[Conv(in_channels=int(256 * width),
                                       out_channels=int(256 * width),
                                       kernel_size=3, stride=1, act=act),
                                  Conv(in_channels=int(256 * width),
                                       out_channels=int(256 * width),
                                       kernel_size=3, stride=1,act=act)]))
            self.cls_pred.append(nn.Conv2d(in_channels=int(256 * width),
                                           out_channels=self.anchors * self.classes,
                                           kernel_size=1,stride=1,padding=0))
            self.reg_pred.append(nn.Conv2d(in_channels=int(256 * width),
                                           out_channels=self.anchors * 4,
                                           kernel_size=1,stride=1,padding=0))
            self.obj_pred.append(nn.Conv2d(in_channels=int(256 * width),
                                           out_channels=self.anchors,
                                           kernel_size=1, stride=1, padding=0))

    def forward(self, *p):
        output = []
        for id, layer in enumerate(p):
            x = self.stems[id](layer)
            cls_x = x
            reg_x = x

            cls_f = self.cls_conv[id](cls_x)
            reg_f = self.reg_conv[id](reg_x)
            cls_out = self.cls_pred[id](cls_f)
            reg_out = self.reg_pred[id](reg_f)
            obj_out = self.obj_pred[id](reg_f)

            output_id = torch.cat([reg_out, obj_out, cls_out], dim=1)

            output_id = torch.flatten(output_id, start_dim=2)
            output_id_split = torch.split(output_id, int(output_id.shape[1] / self.anchors), dim=1)
            output_id = torch.cat(output_id_split, dim=2)
            output.append(output_id)

        output = torch.cat(output, dim=2)
        return output

def build_head(name):
    if name == 'yolov3_head':
        return Yolov3_head
    elif name == 'retina_head':
        return Retina_head
    elif name == ' yolox_head':
        return YOLOX_head
    else:
        raise NotImplementedError('No head named %s'%name)

if __name__ == '__main__':
    input3 = torch.ones([1,128,64,64])
    input4 = torch.ones([1,256,32,32])
    input5 = torch.ones([1,512,16,16])
    # input6 = torch.ones([1,256,8,8])
    # input7 = torch.ones([1,256,4,4])
    head = YOLOX_head(2, 2, 128)
    obj = head(input3,input4,input5)
    print(obj.shape)