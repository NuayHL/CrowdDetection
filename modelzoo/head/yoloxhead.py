import torch
from torch import nn as nn

from modelzoo.common import DWConv, BaseConv

from modelzoo.head.build import HeadRegister

@HeadRegister.register
@HeadRegister.register('yolox_head')
class YOLOX_head(nn.Module):
    def __init__(self, classes, anchors_per_grid, p3_channels=256, depthwise=False, act='silu'):
        super(YOLOX_head, self).__init__()
        self.depthwise = depthwise
        width = p3_channels/256.0
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
            self.stems.append(Conv(in_channels=int(chs),
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

@HeadRegister.register
@HeadRegister.register('yolox_mhead')
class YOLOX_mhead(nn.Module):
    def __init__(self, classes, anchors_per_grid, p3_channels=256, depthwise=False, act='silu'):
        super(YOLOX_mhead, self).__init__()
        self.depthwise = depthwise
        width = p3_channels/256.0
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
            self.stems.append(Conv(in_channels=int(chs),
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

            if self.anchors == 1:
                output_id = torch.cat([reg_out, obj_out, cls_out], dim=1)
            else:
                cls_sp = torch.split(cls_out, int(self.classes), dim=1)
                reg_sp = torch.split(reg_out, 4, dim=1)
                obj_sp = torch.split(reg_out, 1, dim=1)

                output_id = list()

                for i in range(self.anchors):
                    output_id.append(torch.cat([reg_sp[i], obj_sp[i], cls_sp[i]], dim=1))
                output_id = torch.cat(output_id, dim=1)

            output_id = torch.flatten(output_id, start_dim=2)
            output_id_split = torch.split(output_id, int(output_id.shape[1] / self.anchors), dim=1)
            output_id = torch.cat(output_id_split, dim=2)
            output.append(output_id)

        output = torch.cat(output, dim=2)
        return output

@HeadRegister.register
@HeadRegister.register('yolox_head_csp')
class YOLOX_head_csp(nn.Module):
    def __init__(self, classes, anchors_per_grid, p3_channels=256, depthwise=False, act='silu'):
        super(YOLOX_head_csp, self).__init__()
        self.depthwise = depthwise
        width = p3_channels/256.0
        self.act = act
        self.classes = classes
        self.anchors = anchors_per_grid
        self.p3c = p3_channels
        self.in_channels = [int(self.p3c), int(self.p3c * 2), int(self.p3c * 4)]

        self.stems = nn.ModuleList()
        self.cls_conv = nn.ModuleList()
        self.cls_pred = nn.ModuleList()
        self.reg_conv = nn.ModuleList()
        self.loc_conv = nn.ModuleList()
        self.center_pred = nn.ModuleList()
        self.scale_pred = nn.ModuleList()
        self.obj_pred = nn.ModuleList()

        Conv = DWConv if self.depthwise else BaseConv

        for chs in self.in_channels:
            self.stems.append(Conv(in_channels=int(chs),
                                   out_channels=int(256 * width),
                                   kernel_size=1,stride=1, act=self.act))
            self.cls_conv.append(nn.Sequential(
                                *[Conv(in_channels=int(256 * width),
                                       out_channels=int(256 * width),
                                       kernel_size=3, stride=1, act=act),
                                  Conv(in_channels=int(256 * width),
                                       out_channels=int(256 * width),
                                       kernel_size=3, stride=1,act=act)]))
            self.cls_pred.append(nn.Conv2d(in_channels=int(256 * width),
                                           out_channels=self.anchors * self.classes,
                                           kernel_size=1,stride=1,padding=0))

            self.reg_conv.append(Conv(in_channels=int(256 * width),
                                      out_channels=int(256 * width),
                                      kernel_size=3, stride=1, act=act))
            self.loc_conv.append(Conv(in_channels=int(256 * width),
                                      out_channels=int(256 * width),
                                      kernel_size=3, stride=1, act=act))
            self.center_pred.append(nn.Conv2d(in_channels=int(256 * width),
                                              out_channels=self.anchors * 2,
                                              kernel_size=1, stride=1, padding=0))
            self.scale_pred.append(nn.Conv2d(in_channels=int(256 * width),
                                             out_channels=self.anchors * 2,
                                             kernel_size=1, stride=1, padding=0))

            self.obj_pred.append(nn.Sequential(
                *[Conv(in_channels=int(256 * width),
                       out_channels=int(256 * width),
                       kernel_size=3, stride=1, act=act),
                  nn.Conv2d(in_channels=int(256 * width),
                            out_channels=self.anchors,
                            kernel_size=1, stride=1, padding=0)]))
    def forward(self, *p):
        output = []
        for id, layer in enumerate(p):
            x = self.stems[id](layer)
            cls_x = x
            reg_x = x

            cls_f = self.cls_conv[id](cls_x)
            reg_f = self.reg_conv[id](reg_x)

            loc_f = reg_f
            obj_f = reg_f

            cls_out = self.cls_pred[id](cls_f)
            obj_out = self.obj_pred[id](obj_f)

            loc_y = self.loc_conv[id](loc_f)
            center_out = self.center_pred[id](loc_y)
            scale_out = self.scale_pred[id](loc_y)

            output_id = torch.cat([center_out, scale_out, obj_out, cls_out], dim=1)

            output_id = torch.flatten(output_id, start_dim=2)
            output_id_split = torch.split(output_id, int(output_id.shape[1] / self.anchors), dim=1)
            output_id = torch.cat(output_id_split, dim=2)
            output.append(output_id)

        output = torch.cat(output, dim=2)
        return output

