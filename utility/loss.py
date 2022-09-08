import torch
import math
import torch.nn as nn

class GeneralLoss():
    '''
    reg loss: smooth l1
    cls loss: bce + focal
    '''
    def __init__(self, config, device, reduction='sum'):
        self.config = config
        self.device = device
        self.reduction = reduction
        self.loss_parse()
        self.loss_weight = config.loss.weight
        self.zero = torch.tensor(0, dtype=torch.float32).to(device)


    def loss_parse(self):
        self.reg_loss = []
        self.reg_loss_type = self.config.loss.reg_type
        for method in self.reg_loss_type:
            if method in ['ciou', 'diou', 'giou', 'siou']:
                self.reg_loss.append(IOUloss(iou_type=method, bbox_type='xywh', reduction=self.reduction))
            if method in ['l1']:
                self.reg_loss.append(SmoothL1(self.reduction))

        self.cls_loss = FocalBCElogits(self.config, self.device, reduction=self.reduction)

    def __call__(self, dt_list, gt_list):
        '''
        dt_list:[reg_loss,..., cls_loss]
        gt_list:[reg_loss,..., cls_loss]
        '''
        losses = {}
        pos_num_samples = dt_list[0].shape[1]
        # pos_neg_num_samples = dt_list[-1].shape[1]
        losses['cls'] = self.cls_loss(dt_list[-1], gt_list[-1]) * self.loss_weight[-1]

        # cal obj loss independently
        if len(dt_list) - len(self.reg_loss) == 2:
            losses['obj'] = self.cls_loss(dt_list[-2], gt_list[-2]) * self.loss_weight[-2]

        if pos_num_samples != 0:
            for loss, loss_name, loss_weight, reg_dt, reg_gt in \
                    zip(self.reg_loss, self.reg_loss_type, self.loss_weight, dt_list, gt_list):
                losses[loss_name] = loss(reg_dt,reg_gt) * loss_weight
        else:
            for loss_name in self.reg_loss_type:
                losses[loss_name] = self.zero
        fin_loss = 0
        for loss in losses.values():
            fin_loss += loss
        for key in losses:
            losses[key] = losses[key].detach().cpu().item()
        return fin_loss, losses

    # def __call__(self, cls_dt, reg_dt, obj_dt, cls_gt, reg_gt, obj_gt):
    #     '''
    #     dt: The input dt should be the same format as assign result
    #         i.e.  Tensor: Batchsize X (4+1+classes) X samples
    #     gt: list, each is np.array, with shape (4+1) at dim=-1
    #     '''
    #     losses = {}
    #     pos_num_samples = reg_gt.shape[1]
    #     pos_neg_num_samples = obj_gt.shape[0]
    #     losses['obj'] = self.obj_loss(obj_dt, obj_gt) / pos_neg_num_samples * self.loss_weight[2]
    #     # if assignment return no positive object
    #     if pos_num_samples != 0:
    #         for i, loss in enumerate(self.reg_loss):
    #             losses[self.reg_loss_type[i]] = loss(reg_dt, reg_gt) * self.loss_weight[0] / pos_num_samples
    #             if self.reg_loss_type[i] == 'l1':
    #                 losses['l1'] /= self.l1_coe
    #
    #         losses['cls'] = self.cls_loss(cls_dt, cls_gt)/pos_neg_num_samples/self.config.data.numofclasses*self.loss_weight[1]
    #     else:
    #         for loss_name in self.reg_loss_type:
    #             losses[loss_name] = self.zero
    #         losses['cls'] = self.zero
    #     fin_loss = 0
    #     for loss in losses.values():
    #         loss.clamp_(0, 100)
    #         fin_loss += loss
    #     for key in losses:
    #         losses[key] = losses[key].detach().cpu().item()
    #     return fin_loss, losses

class FocalBCElogits():
    def __init__(self, config, device, reduction='none'):
        self.reduction = reduction
        self.use_focal = config.loss.use_focal
        self.alpha = config.loss.focal_alpha
        self.gamma = config.loss.focal_gamma
        self.device = device
        self.baseloss = nn.BCEWithLogitsLoss(reduction='none')
    def __call__(self, dt_cls, gt_cls):
        bceloss = self.baseloss(dt_cls, gt_cls)
        if self.use_focal:
            alpha = torch.ones(dt_cls.shape).to(self.device) * self.alpha
            alpha = torch.where(torch.eq(gt_cls, 1.), alpha, 1. - alpha)
            focal_weight = torch.where(torch.eq(gt_cls, 1.), 1 - dt_cls, dt_cls)
            focal_weight = alpha * torch.pow(focal_weight, self.gamma)
            bceloss *= focal_weight
        if self.reduction == 'sum':
            return bceloss.sum()
        elif self.reduction == 'mean':
            return bceloss.mean()
        else:
            return bceloss

class FocalBCE():
    def __init__(self, config, device):
        self.use_focal = config.loss.use_focal
        self.alpha = config.loss.focal_alpha
        self.gamma = config.loss.focal_gamma
        self.device = device
        self.baseloss = BCElossAmp(reduction='none')
    def __call__(self, dt_cls, gt_cls):
        bceloss = self.baseloss(dt_cls, gt_cls)
        if self.use_focal:
            alpha = torch.ones(dt_cls.shape).to(self.device) * self.alpha
            alpha = torch.where(torch.eq(gt_cls, 1.), alpha, 1. - alpha)
            focal_weight = torch.where(torch.eq(gt_cls, 1.), 1 - dt_cls, dt_cls)
            focal_weight = alpha * torch.pow(focal_weight, self.gamma)
            bceloss *= focal_weight
        return bceloss.sum()

class BCElossAmp():
    def __init__(self, reduction='none', eps=1e-5):
        self.eps = eps
        self.reduction = reduction
    def __call__(self, dt, gt):
        dt = dt.to(torch.float32)
        dt = dt.clamp(self.eps, 1-self.eps)
        gt = gt.clamp(0, 1)
        loss = - gt * torch.log(dt) + (gt - 1.0) * torch.log(1.0 - dt)
        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss

class SmoothL1():
    def __init__(self, reduction='sum'):
        self.baseloss = nn.SmoothL1Loss(beta=1./9, reduction=reduction)
    def __call__(self, dt, gt):
        return self.baseloss(dt, gt)

class IOUloss():
    """ Calculate IoU loss.
        based on https://github.com/meituan/YOLOv6/blob/main/yolov6/models/loss.py
    """
    def __init__(self, bbox_type='xywh', iou_type='ciou', reduction='none', eps=1e-7):
        """ Setting of the class.
        Args:
            bbox_type: (string), must be one of 'xywh' or 'x1y1x2y2' or 'x1y1wh'.
            iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid devide by zero error.
        """
        self.box_format = bbox_type
        self.iou_type = iou_type.lower()
        self.reduction = reduction
        self.eps = eps

    def __call__(self, box1, box2):
        """ calculate iou. box1 and box2 are torch tensor with shape [4, m] and [4, m].
        """
        if self.box_format == 'x1y1x2y2':
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        elif self.box_format == 'xywh':
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        elif self.box_format == 'x1y1wh':
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
        else:
            raise RuntimeError("None support bbox_type: %s"%self.box_format)
        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union = w1 * h1 + w2 * h2 - inter + self.eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if self.iou_type == 'giou':
            c_area = cw * ch + self.eps  # convex area
            iou = iou - (c_area - union) / c_area
        elif self.iou_type in ['diou', 'ciou']:
            c2 = cw ** 2 + ch ** 2 + self.eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if self.iou_type == 'diou':
                iou = iou - rho2 / c2
            elif self.iou_type == 'ciou':
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + self.eps))
                iou = iou - (rho2 / c2 + v * alpha)
        elif self.iou_type == 'siou':
            # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + self.eps
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            rho_x_g = torch.clamp(gamma * rho_x, max = 50)
            rho_y_g = torch.clamp(gamma * rho_y, max = 50)
            distance_cost = 2 - torch.exp(rho_x_g) - torch.exp(rho_y_g)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
        loss = 1.0 - iou

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

def updata_loss_dict(ori_dict, target_dict):
    for key in target_dict:
        if key not in ori_dict:
            ori_dict[key] = target_dict[key]
        else:
            ori_dict[key] += target_dict[key]

def loss_dict_to_str(loss_dict):
    fin_str = '|| '
    for key in loss_dict:
        fin_str += str(key) + ': % 6.2f'%loss_dict[key]+ ' || '
    return fin_str