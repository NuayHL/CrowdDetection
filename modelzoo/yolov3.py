import torch
import torch.nn as nn

from utility.assign import AnchorAssign
from utility.anchors import generateAnchors
from utility.loss import GeneralLoss

class Yolov3(nn.Module):
    def __init__(self, config, backbone, neck, head):
        super(Yolov3, self).__init__()
        self.config = config
        self.pretrain_settings()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, sample):
        if self.training:
            return self.training_loss(sample)
        else:
            return self.inferencing(sample)
    def core(self,input):
        p3, p4, p5 = self.backbone(input)
        p3, p4, p5 = self.neck(p3, p4, p5)
        p3, p4, p5 = self.head(p3, p4, p5)
        return p3, p4, p5

    def set(self, args, device):
        self.device = device
        self.anchors_per_grid = len(self.config.model.anchor_ratios) * len(self.config.model.anchor_scales)
        self.assignment = AnchorAssign(self.config, device)
        self.loss = GeneralLoss(self.config, device)

    def training_loss(self,sample):
        p3, p4, p5 = self.core(sample['imgs'])
        dt = self._result_parse((p3,p4,p5))

        reg_dt = dt[:, :4,:]
        obj_dt = dt[:, 4, :]
        cls_dt = dt[:, 5:, :]

        for anns_ib in sample['annss']:
            assign_result = self.assignment.assign(anns_ib)
            label_pos_neg


        if self.useignore:
            assign_result, gt = self.label_assignment.assign(gt)
        else:
            assign_result = self.label_assignment.assign(gt)

        cls_dt = torch.clamp(cls_dt, 1e-7, 1.0 - 1e-7)

        if torch.cuda.is_available():
            cls_dt = cls_dt.to(self.device)
            reg_dt = reg_dt.to(self.device)

        # positive: exclude ignored sample
        # assigned: positive sample
        for ib in range(self.batch_size):
            positive_idx_cls = torch.ge(assign_result[ib], -0.1)
            # the not ignored ones
            positive_idx_box = torch.ge(assign_result[ib] - 1.0, -0.1)
            # the assigned ones
            debug_sum_po = positive_idx_box.sum()

            imgAnn = gt[ib]
            if not self.useignore:
                imgAnn = torch.from_numpy(imgAnn).float()
                if torch.cuda.is_available():
                    imgAnn = imgAnn.to(self.device)

            assign_result_box = assign_result[ib][positive_idx_box].long() - 1
            target_anns = imgAnn[assign_result_box]

            # cls loss
            one_hot_bed = torch.zeros((assign_result.shape[1], self.classes), dtype=torch.int64)
            if torch.cuda.is_available():
                one_hot_bed = one_hot_bed.to(self.device)

            one_hot_bed[positive_idx_box, target_anns[:, 4].long() - 1] = 1

            assign_result_cal = one_hot_bed[positive_idx_cls]
            debug_sum_ = assign_result_cal.sum()
            cls_dt_cal = cls_dt[ib, :, positive_idx_cls].t()

            cls_loss_ib = - assign_result_cal * torch.log(cls_dt_cal) + \
                          (assign_result_cal - 1.0) * torch.log(1.0 - cls_dt_cal)

            debug_sum_ib = cls_loss_ib.sum()

            if self.usefocal:
                if torch.cuda.is_available():
                    alpha = torch.ones(cls_dt_cal.shape).to(self.device) * self.alpha
                else:
                    alpha = torch.ones(cls_dt_cal.shape) * self.alpha

                alpha = torch.where(torch.eq(assign_result_cal, 1.), alpha, 1. - alpha)
                focal_weight = torch.where(torch.eq(assign_result_cal, 1.), 1 - cls_dt_cal, cls_dt_cal)
                focal_weight = alpha * torch.pow(focal_weight, self.gamma)
                cls_fcloss_ib = focal_weight * cls_loss_ib
            else:
                cls_fcloss_ib = cls_loss_ib

            cls_loss.append(cls_fcloss_ib.sum() / positive_idx_box.sum())

            # bbox loss
            anch_w_box = self.anch_w[positive_idx_box]
            anch_h_box = self.anch_h[positive_idx_box]
            anch_x_box = self.anch_x[positive_idx_box]
            anch_y_box = self.anch_y[positive_idx_box]

            reg_dt_assigned = reg_dt[ib, :, positive_idx_box]

            dt_bbox_x = anch_x_box + reg_dt_assigned[0, :] * anch_w_box
            dt_bbox_y = anch_y_box + reg_dt_assigned[1, :] * anch_h_box
            reg_dt_assigned_wh = torch.clamp(reg_dt_assigned[2:, :], max=50)
            dt_bbox_w = anch_w_box * torch.exp(reg_dt_assigned_wh[0, :])
            dt_bbox_h = anch_h_box * torch.exp(reg_dt_assigned_wh[1, :])

            dt_bbox = torch.stack([dt_bbox_x, dt_bbox_y, dt_bbox_w, dt_bbox_h])

            target_anns[:, 0] += 0.5 * target_anns[:, 2]
            target_anns[:, 1] += 0.5 * target_anns[:, 3]

            box_regression_loss_ib = self.iouloss(dt_bbox, target_anns.t())

            bbox_loss.append(box_regression_loss_ib / positive_idx_box.sum())

        bbox_loss = torch.stack(bbox_loss)
        cls_loss = torch.stack(cls_loss)
        bbox_loss = bbox_loss.sum()
        cls_loss = cls_loss.sum()
        # print('cls loss:%.8f'%cls_loss, 'bbox loss:%.4f'%bbox_loss)
        loss = torch.add(bbox_loss, cls_loss)
        return loss / self.batch_size
    def inferencing(self, sample):
        pass

    def _result_parse(self, triple):
        '''
        flatten the results according to the format of anchors
        '''
        out = torch.zeros((triple[0].shape[0], int(5 + self.config.data.numofclasses), 0))
        if torch.cuda.is_available():
            out = out.to(self.device)
        for fp in triple:
            fp = torch.flatten(fp, start_dim=2)
            split = torch.split(fp, int(fp.shape[1] / self.anchors_per_grid), dim=1)
            out = torch.cat([out]+list(split), dim=2)
        return out


