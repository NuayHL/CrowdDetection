import torch
from torch import nn as nn

from modelzoo.head.build import HeadRegister
from modelzoo.common import YOLOv7_common
Conv = YOLOv7_common.Conv

@HeadRegister.register
@HeadRegister.register('yolov7_head')
class Yolov7_head(nn.Module):
    def __init__(self, classes, anchors_per_grid, p3_channels):
        super(Yolov7_head, self).__init__()
        self.anchors = anchors_per_grid
        conv = YOLOv7_common.RepConv
        transition_channels = 32
        output_ch = (4 + 1 + classes) * anchors_per_grid
        assert p3_channels == 128
        self.rep_conv_1 = conv(transition_channels * 4, transition_channels * 8, 3, 1)
        self.rep_conv_2 = conv(transition_channels * 8, transition_channels * 16, 3, 1)
        self.rep_conv_3 = conv(transition_channels * 16, transition_channels * 32, 3, 1)

        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, output_ch, 1)
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, output_ch, 1)
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, output_ch, 1)

    def forward(self, P3, P4, P5):
        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)

        p3 = self.yolo_head_P3(P3)
        p4 = self.yolo_head_P4(P4)
        p5 = self.yolo_head_P5(P5)

        return self._result_parse((p3, p4, p5))

    def _result_parse(self, triple):
        """
        flatten the results according to the format of anchors
        """
        out = []
        for fp in triple:
            fp = torch.flatten(fp, start_dim=2)
            split = torch.split(fp, int(fp.shape[1] / self.anchors), dim=1)
            fp = torch.cat(split, dim=2)
            out.append(fp)
        out = torch.cat(out,dim=2)
        return out