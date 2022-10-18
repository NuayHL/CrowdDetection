import torch
import torch.nn as nn

from modelzoo.common import DWConv, BaseConv

from modelzoo.head.build import HeadRegister

@HeadRegister.register
@HeadRegister.register('pdhead')
class PDHead(nn.Module):
    def __init__(self, classes, anchors_per_grid, p3c, width=1.0, act='silu'):
        super(PDHead, self).__init__()
        self.act = act
        self.classes = classes
        self.anchors = anchors_per_grid
        self.p3c = p3c
        self.in_channels = [int(self.p3c), int(self.p3c * 2), int(self.p3c * 4)]

        self.stems = nn.ModuleList()
        self.reg_conv = nn.ModuleList()
        self.reg_pred = nn.ModuleList()
        self.obj_pred = nn.ModuleList()

        Conv = BaseConv

        for chs in self.in_channels:
            self.stems.append(Conv(in_channels=int(chs),
                                   out_channels=int(256 * width),
                                   kernel_size=1,stride=1, act=self.act))
            self.reg_conv.append(nn.Sequential(
                                *[Conv(in_channels=int(256 * width),
                                       out_channels=int(256 * width),
                                       kernel_size=3, stride=1, act=act),
                                  Conv(in_channels=int(256 * width),
                                       out_channels=int(256 * width),
                                       kernel_size=3, stride=1,act=act)]))
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
            reg_x = x

            reg_f = self.reg_conv[id](reg_x)
            reg_out = self.reg_pred[id](reg_f)
            obj_out = self.obj_pred[id](reg_f)

            output_id = torch.cat([reg_out, obj_out], dim=1)

            output_id = torch.flatten(output_id, start_dim=2)
            output_id_split = torch.split(output_id, int(output_id.shape[1] / self.anchors), dim=1)
            output_id = torch.cat(output_id_split, dim=2)
            output.append(output_id)

        output = torch.cat(output, dim=2)
        return output

@HeadRegister.register
@HeadRegister.register('pdhead_csp')
class PDHead_csp(nn.Module):
    def __init__(self, classes, anchors_per_grid, p3c, width=1.0, act='silu'):
        super(PDHead_csp, self).__init__()
        self.act = act
        self.classes = classes
        self.anchors = anchors_per_grid
        self.p3c = p3c
        self.in_channels = [int(self.p3c), int(self.p3c * 2), int(self.p3c * 4)]

        self.stems = nn.ModuleList()
        self.reg_conv = nn.ModuleList()
        self.loc_conv = nn.ModuleList()
        self.center_pred = nn.ModuleList()
        self.scale_pred = nn.ModuleList()
        self.obj_pred = nn.ModuleList()

        Conv = BaseConv

        for chs in self.in_channels:
            self.stems.append(Conv(in_channels=int(chs),
                                   out_channels=int(256 * width),
                                   kernel_size=1,stride=1, act=self.act))
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
            x = self.reg_conv[id](x)
            loc_x = x
            obj_x = x

            obj_out = self.obj_pred[id](obj_x)
            loc_y = self.loc_conv[id](loc_x)
            center_out = self.center_pred[id](loc_y)
            scale_out = self.scale_pred[id](loc_y)

            output_id = torch.cat([center_out, scale_out, obj_out], dim=1)

            output_id = torch.flatten(output_id, start_dim=2)
            output_id_split = torch.split(output_id, int(output_id.shape[1] / self.anchors), dim=1)
            output_id = torch.cat(output_id_split, dim=2)
            output.append(output_id)

        output = torch.cat(output, dim=2)
        return output
