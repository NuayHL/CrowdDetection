import time
import torch
import torchvision
import numpy as np
import warnings

class NMS():
    def __init__(self, config):
        self.config = config
        self.nms_type = config.inference.nms_type.lower()
        self.conf_thres = config.inference.obj_thres
        self.iou_thres = config.inference.iou_thres
        self.maxwh = max(config.data.input_width, config.data.input_height)
        self.set_func()

    def set_func(self):
        if self.nms_type == "nms":
            self.func = self.standard
        elif self.nms_type in ["soft_n", "soft"]:
            self.func = self.normal_kernel
        elif self.nms_type == "soft_g":
            self.func = self.gaussian_kernel
        else:
            raise NotImplementedError("Type %s for NMS not supported, please using \'nms\' for standard NMS type,"
                                      " \'soft\' for soft NMS. Check nms.py for more options." % self.nms_type)

    def __call__(self, dets, class_indepent=False):
        '''Input: dets.shape = [B, n, xywh+o+c] reshaped raw output of the model'''
        num_classes = dets.shape[2] - 5  # number of classes
        batch_size = dets.shape[0]
        pred_candidates = dets[..., 4] > self.conf_thres  # candidates
        output = [torch.zeros((0, 6), device=dets.device)] * batch_size
        for ib, det in enumerate(dets):
            det = det[pred_candidates[ib]]
            det[:, 5:] *= det[:, 4:5]
            conf, categories = det[:, 5:].max(dim=1, keepdim=True)
            class_offset = categories.float() * (0 if not class_indepent else self.maxwh)
            box = xywh2xyxy(det[:, :4]) + class_offset
            det = torch.cat([box, conf, categories], dim=1)
            if det.shape[0] == 0:
                continue
            kept_box_mask = self._nms(det[:, :5])
            output[ib] = det[kept_box_mask]
        return output

    def cal_block_output(self, dets, indicator, class_indepent=False):
        '''Input: dets.shape = [B, n, xywh+o+c] reshaped raw output of the model'''
        num_classes = dets.shape[2] - 5  # number of classes
        batch_size = dets.shape[0]
        pred_candidates = dets[..., 4] > self.conf_thres  # candidates
        output_indicator = indicator[pred_candidates]
        output = [None] * batch_size
        for ib, det in enumerate(dets):
            det = det[pred_candidates[ib]]
            det[:, 5:] *= det[:, 4:5]
            conf, categories = det[:, 5:].max(dim=1, keepdim=True)
            class_offset = categories.float() * (0 if not class_indepent else self.maxwh)
            box = xywh2xyxy(det[:, :4]) + class_offset
            det = torch.cat([box, conf, categories], dim=1)
            if det.shape[0] == 0:
                continue
            kept_box_mask = self._nms(det[:, :5])
            output[ib] = [kept_box_mask]

        real_output_indicator = [single_indicator[pred_candidate][kept_box]
                                 for single_indicator, kept_box, pred_candidate
                                 in zip(indicator, output, pred_candidates)]
        return output_indicator, real_output_indicator

    def _nms(self, dets):
        '''det: [n,5], 5: x1y1x2y2 score, return kept indices. Warning: n must > 0'''
        eps = 1e-8
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = dets[:, 4].sort().indices.flip(0)
        kept_output = []

        zero = torch.tensor(0.0).to(dets.device)
        one = torch.tensor(1.0).to(dets.device)

        while order.shape[0] > 0:
            pick_ind = order[0]
            kept_output.append(pick_ind)
            order = order[1:]
            xx1 = torch.maximum(x1[pick_ind], x1[order])
            yy1 = torch.maximum(y1[pick_ind], y1[order])
            xx2 = torch.minimum(x2[pick_ind], x2[order])
            yy2 = torch.minimum(y2[pick_ind], y2[order])

            inter = torch.maximum(zero, xx2 - xx1) * torch.maximum(zero, yy2 - yy1)
            iou = inter / (areas[pick_ind] + areas[order] - inter + eps)

            weight = torch.where(iou > self.iou_thres, self.func(iou), one)
            dets[order, 4] *= weight

            order = order[torch.ge(dets[order, 4], self.conf_thres)]

        kept_output = torch.stack(kept_output, dim=0)
        return kept_output

    @staticmethod
    def standard(iou):
        return 0

    @staticmethod
    def normal_kernel(iou):
        return 1 - iou

    @staticmethod
    def gaussian_kernel(iou):
        return torch.exp(-(iou * iou) / 0.5)

class SetNMS(NMS):
    def __init__(self, config):
        super(SetNMS, self).__init__(config)
        self.mip_k = int(self.config.model.assignment_extra[0]['k'])
        self.set_table = self.gen_set_table()

    def __call__(self, dets, class_indepent=False):
        '''Input: dets.shape = [B, n, xywh+o+c] reshaped raw output of the model'''
        num_classes = dets.shape[2] - 5  # number of classes
        batch_size = dets.shape[0]
        pred_candidates = dets[..., 4] > self.conf_thres  # candidates
        output = [torch.zeros((0, 6), device=dets.device)] * batch_size
        for ib, det in enumerate(dets):
            set_table = self.set_table.to(det.device).clone().unsqueeze(dim=1)
            det = det[pred_candidates[ib]]
            set_table_ib = set_table[pred_candidates[ib]]
            det[:, 5:] *= det[:, 4:5]
            conf, categories = det[:, 5:].max(dim=1, keepdim=True)
            class_offset = categories.float() * (0 if not class_indepent else self.maxwh)
            box = xywh2xyxy(det[:, :4]) + class_offset
            det = torch.cat([box, conf, categories], dim=1)
            if det.shape[0] == 0:
                continue
            det_with_set = torch.cat((det[:, :5], set_table_ib), dim=1)
            kept_box_mask = self._nms(det_with_set)
            output[ib] = det[kept_box_mask]
        return output

    def _nms(self, dets):
        '''det: [n,5], 5: x1y1x2y2 score, return kept indices. Warning: n must > 0'''
        eps = 1e-8
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        set_indexes = dets[:, 5].int()

        areas = (x2 - x1) * (y2 - y1)
        order = dets[:, 4].sort().indices.flip(0)
        kept_output = []

        zero = torch.tensor(0.0).to(dets.device)
        one = torch.tensor(1.0).to(dets.device)

        while order.shape[0] > 0:
            pick_ind = order[0]
            order = order[1:]

            kept_output.append(pick_ind)
            set_index = set_indexes[pick_ind]
            left_order_set = set_indexes[order]

            xx1 = torch.maximum(x1[pick_ind], x1[order])
            yy1 = torch.maximum(y1[pick_ind], y1[order])
            xx2 = torch.minimum(x2[pick_ind], x2[order])
            yy2 = torch.minimum(y2[pick_ind], y2[order])

            inter = torch.maximum(zero, xx2 - xx1) * torch.maximum(zero, yy2 - yy1)
            iou = inter / (areas[pick_ind] + areas[order] - inter + eps)

            weight = torch.ones_like(iou)

            pos_mask = torch.le(iou, self.iou_thres)
            set_mask = torch.eq(left_order_set, set_index)

            fin_mask = ~torch.logical_or(pos_mask, set_mask)
            weight[fin_mask] = self.func(iou[fin_mask])
            dets[order, 4] *= weight

            order = order[torch.ge(dets[order, 4], self.conf_thres)]

        kept_output = torch.stack(kept_output, dim=0)
        return kept_output

    def gen_set_table(self):
        from utility.anchors import Anchor
        used_anchor = Anchor(self.config)
        num_in_each_level = torch.tensor(used_anchor.get_num_in_each_level())
        anchor_per_grid = used_anchor.get_anchors_per_grid()
        ori_num_anchor = int(num_in_each_level.sum().item())
        total_num_anchor = ori_num_anchor * self.mip_k

        size_in_each_level = (num_in_each_level / anchor_per_grid).int()

        ori_table = torch.arange(ori_num_anchor).int()
        set_table = torch.zeros(total_num_anchor).int()

        set_i = 0
        ori_i = 0
        for level_size in size_in_each_level:
            for _ in range(anchor_per_grid):
                for i in range(self.mip_k):
                    set_table[set_i: set_i+level_size] = ori_table[ori_i: ori_i+level_size]
                    set_i += level_size
                ori_i += level_size
        assert set_i == total_num_anchor
        return set_table

if __name__ == "__main__":
    # num_in_each_level = torch.tensor([8,2])
    # anchor_per_grid = 2
    # mip_k = 3
    pass

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes. 5: xywh + objectness
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """

    num_classes = prediction.shape[2] - 5  # number of classes
    pred_candidates = prediction[..., 4] > conf_thres  # candidates

    # Check the parameters.
    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)
        else:  # Only keep the class with highest scores.
            conf, class_idx = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + class_offset, x[:, 4:5]  # boxes (offset by class), scores
        #keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        keep_box_idx = nms_tensor(torch.cat([boxes, scores], dim=1), iou_thres)
        #keep_box_idx = soft_nms_tensor(torch.cat([boxes, scores], dim=1), iou_thres, conf_thres)
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]
        if (time.time() - tik) > time_limit:
            print(f'WARNING: NMS cost time exceed the limited {time_limit}s.')
            break  # time limit exceeded

    return output

def nms_tensor(dets: torch.Tensor, iou_thres=0.45):
    '''det: [n,5], 5: x1y1x2y2 score, return kept indices'''
    eps = 1e-8
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = dets[:, 4].sort().indices.flip(0)
    kept_output = []

    zero = torch.tensor(0).to(dets.device)

    while order.shape[0] > 0:
        pick_ind = order[0]
        kept_output.append(pick_ind)
        order = order[1:]
        xx1 = torch.maximum(x1[pick_ind], x1[order])
        yy1 = torch.maximum(y1[pick_ind], y1[order])
        xx2 = torch.minimum(x2[pick_ind], x2[order])
        yy2 = torch.minimum(y2[pick_ind], y2[order])

        inter = torch.maximum(zero, xx2 - xx1) * torch.maximum(zero, yy2 - yy1)
        iou = inter / (areas[pick_ind] + areas[order] - inter + eps)
        order = order[torch.le(iou, iou_thres)]

    kept_output = torch.stack(kept_output, dim=0)
    return kept_output

def soft_nms_tensor(dets: torch.Tensor, iou_thres=0.45, score_thres=0.5):
    '''det: [n,5], 5: x1y1x2y2 score, return kept indices'''
    eps = 1e-8
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = dets[:, 4].sort().indices.flip(0)
    kept_output = []

    zero = torch.tensor(0.0).to(dets.device)
    one = torch.tensor(1.0).to(dets.device)

    def standard(iou):
        return 0
    def normal_kernel(iou):
        return 1-iou
    def gaussian_kernel(iou):
        return torch.exp(-(iou*iou)/0.5)

    func = normal_kernel

    while order.shape[0] > 0:
        pick_ind = order[0]
        kept_output.append(pick_ind)
        order = order[1:]
        xx1 = torch.maximum(x1[pick_ind], x1[order])
        yy1 = torch.maximum(y1[pick_ind], y1[order])
        xx2 = torch.minimum(x2[pick_ind], x2[order])
        yy2 = torch.minimum(y2[pick_ind], y2[order])

        inter = torch.maximum(zero, xx2 - xx1) * torch.maximum(zero, yy2 - yy1)
        iou = inter / (areas[pick_ind] + areas[order] - inter + eps)

        weight = torch.where(iou>iou_thres, func(iou), one)
        dets[order, 4] *= weight

        order = order[torch.ge(dets[order, 4], score_thres)]

    kept_output = torch.stack(kept_output, dim=0)
    return kept_output


def nms_np(dets, iou_thresh):
    '''det: [n,5], 5: x1y1x2y2 score, return kept indices'''
    eps = 1e-8
    x1 = np.ascontiguousarray(dets[:, 0])
    y1 = np.ascontiguousarray(dets[:, 1])
    x2 = np.ascontiguousarray(dets[:, 2])
    y2 = np.ascontiguousarray(dets[:, 3])

    areas = (x2 - x1) * (y2 - y1)
    order = dets[:, 4].argsort()[::-1]
    kept_output = []

    while order.size > 0:
        pick_ind = order[0]
        kept_output.append(pick_ind)

        xx1 = np.maximum(x1[pick_ind], x1[order[1:]])
        yy1 = np.maximum(y1[pick_ind], y1[order[1:]])
        xx2 = np.minimum(x2[pick_ind], x2[order[1:]])
        yy2 = np.minimum(y2[pick_ind], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[pick_ind] + areas[order[1:]] - inter + eps)
        order = order[np.where(iou <= iou_thresh)[0] + 1] # +1 since the idxs is cal at order[1:]

    return kept_output

def xywh2xyxy(x):
    # Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
