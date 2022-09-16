#this .py is for Iou related algorithms

import torch
import torch.nn as nn
from copy import deepcopy

class IOU(nn.Module):
    def __init__(self,dt_type="x1y1x2y2", gt_type="x1y1x2y2", ioutype="iou"):
        '''
        bboxtype: x1y1x2y2/x1y1wh/xywh
        default using x1y1x2y2 for calculation
        '''
        super(IOU, self).__init__()
        self.ioutype = ioutype.lower()
        self.dt_type = dt_type
        self.gt_type = gt_type

    @torch.no_grad()
    def forward(self,dt,gt):
        '''
        WARNING: the input must be bboxs, i.e. len(dt.shape)==2
        WARNING: the input must be torch.Tensor
        :param dt: detect bboxes or anchor bboxes
        :param gt: gt bboxes
        :return: ious
        '''
        dt, gt = self._transfer(dt, gt)

        if self.ioutype == "iou":
            return self._iou(dt, gt)
        if self.ioutype == "giou":
            return self._giou(dt,gt)
        else:
            raise NotImplementedError("Unknown iouType")

    def _transfer(self,dt,gt):
        if self.dt_type == "x1y1wh":
            dt = self._x1y1wh_to_x1y1x2y2(dt)
        elif self.dt_type == "xywh":
            dt = self._xywh_to_x1y1x2y2(dt)
        elif self.dt_type == "x1y1x2y2":
            pass
        else:
            raise NotImplementedError("Unknown inputType")

        if self.gt_type == "x1y1wh":
            gt = self._x1y1wh_to_x1y1x2y2(gt)
        elif self.gt_type == "xywh":
            gt = self._xywh_to_x1y1x2y2(gt)
        elif self.gt_type == "x1y1x2y2":
            pass
        else:
            raise NotImplementedError("Unknown gtType")

        return dt, gt

    def _iou(self, a, b):
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        w_int = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
        h_int = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

        w_int = torch.clamp(w_int, min=0)
        h_int = torch.clamp(h_int, min=0)

        intersection = w_int * h_int

        union = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - intersection

        union = torch.clamp(union, min=1e-8)

        IoU = intersection / union

        return IoU

    def _giou(self, a, b):
        w_int = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
        h_int = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
        w_int = torch.clamp(w_int, min = 0)
        h_int = torch.clamp(h_int, min = 0)

        w_co = torch.max(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.min(torch.unsqueeze(a[:, 0], 1), b[:, 0])
        h_co = torch.max(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.min(torch.unsqueeze(a[:, 1], 1), b[:, 1])
        w_co = torch.clamp(w_co, min = 0)
        h_co = torch.clamp(h_co, min = 0)
        pass

    def _x1y1wh_to_x1y1x2y2(self, input):
        input_ = input.clone()
        input_[:, 2] = input[:, 0] + input[:, 2]
        input_[:, 3] = input[:, 1] + input[:, 3]
        return input_

    def _xywh_to_x1y1x2y2(self,input):
        input_ = input.clone()
        input_[:, 0] = input[:,0] - 0.5 * input[:,2]
        input_[:, 1] = input[:,1] - 0.5 * input[:,3]
        input_[:, 2] = input_[:,0] + input[:,2]
        input_[:, 3] = input_[:,1] + input[:,3]
        return input_
