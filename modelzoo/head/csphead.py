import torch
import torch.nn as nn

class CenterScalehead(nn.Module):
    def __init__(self, classes, anchors_per_grid, p3c, depth=1.0, act='silu'):
        super(CenterScalehead, self).__init__()
        self.act = act
        self.classes = classes
        self.anchors = anchors_per_grid
        self.p3c = p3c
        self.in_channels = [int(self.p3c), int(self.p3c * 2), int(self.p3c * 4)]

        self.stems = nn.ModuleList()
        self.cls_conv = nn.ModuleList()

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