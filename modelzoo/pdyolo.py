import torch
import torch.nn as nn


from modelzoo.basemodel import BaseODModel
from utility.assign import get_assign_method
from utility.anchors import result_parse, Anchor
from utility.loss import PedestrianLoss
from utility.nms import NMS
from utility.result import Result


class PDYOLO(BaseODModel):
    '''4 + 1'''

    def __init__(self, config, backbone, neck, head):
        super(PDYOLO, self).__init__()
        self.config = config
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.sigmoid = nn.Sigmoid()
        self.input_shape = (self.config.data.input_width, self.config.data.input_height)
        self.nms = NMS(config)

    def core(self, input):
        fms = self.backbone(input)
        rea_fms = self.neck(*fms)
        dt = self.head(*rea_fms)
        return dt

    def set(self, args, device):
        self.device = device
        self.anchors_per_grid = len(self.config.model.anchor_ratios) * len(self.config.model.anchor_scales)
        self.assignment = get_assign_method(self.config, device)
        self.assign_type = self.config.model.assignment_type.lower()
        assert self.assign_type not in ['simota'], \
            'PDYOLO does not support simota assignment, consider default or pdsimota'
        self.anch_gen = Anchor(self.config)
        self.use_anchor = self.config.model.use_anchor
        if self.use_anchor:
            self.anchs = torch.from_numpy(self.anch_gen.gen_Bbox(singleBatch=True)).float().to(device)
        else:
            assert self.assign_type != "default", "Assign type %s do not support anchor free style" % self.assign_type
            self.anchs = torch.from_numpy(self.anch_gen.gen_points(singleBatch=True)).float().to(device)
            scale = self.config.model.stride_scale
            single_stride = torch.from_numpy(self.anch_gen.gen_stride(singleBatch=True) * scale).float().to(
                device).unsqueeze(1)
            self.stride = torch.cat([single_stride, single_stride], dim=1)
        self.num_of_proposal = self.anchs.shape[0]

        self.use_l1 = True if 'l1' in self.config.loss.reg_type else False

        assert True in ['iou' in loss for loss in self.config.loss.reg_type], 'Please adding iou-type loss'
        self.loss = PedestrianLoss(self.config, device, reduction='mean')

    def training_loss(self, sample):
        dt = self.core(sample['imgs'])  # [B, 4+1, n_anchors_all]
        obj_dt = dt[:, 4, :].clone()  # [B, n_anchors_all]
        ori_reg_dt = dt[:, :4, :]  # [B, 4, n_anchors_all]
        shift_dt = self.get_shift_bbox(ori_reg_dt)

        # with amp.autocast(enabled=False):
        with torch.no_grad():
            if self.assign_type == 'pdsimota':
                assign_result, gt, weight = self.assignment.assign(sample['annss'],
                                                                   torch.cat([shift_dt, obj_dt.unsqueeze(1)],
                                                                             dim=1))
            else:
                assign_result, gt, weight = self.assignment.assign(sample['annss'])

        pos_mask = []

        obj_gt = []
        shift_gt = []
        if self.use_l1:
            l1_gt = []

        for ib in range(len(gt)):
            assign_result_ib, gt_ib = assign_result[ib], gt[ib]
            pos_mask_ib = torch.gt(assign_result_ib, 0.5)
            pos_mask.append(pos_mask_ib)

            label_pos_generate = (assign_result_ib[pos_mask_ib] - 1).long()

            shift_gt_ib = gt_ib[label_pos_generate, :4].t()
            shift_gt.append(shift_gt_ib)
            if self.use_l1:
                l1_gt_ib = gt_ib[label_pos_generate, :4]
                l1_gt_ib = self.get_l1_target(l1_gt_ib, pos_mask_ib)
                l1_gt.append(l1_gt_ib)
            obj_gt_ib = assign_result_ib.clamp(0, 1)
            if weight:
                obj_gt_ib[pos_mask_ib] *= weight[ib]
            obj_gt.append(obj_gt_ib)

        pos_mask = torch.cat(pos_mask, dim=0)
        dt = dict()
        gt = dict()
        dt['obj'] = obj_dt.view(-1)
        gt['obj'] = torch.cat(obj_gt, dim=0)
        dt['iou'] = shift_dt.permute(1, 0, 2).reshape(4, -1)[:, pos_mask]
        gt['iou'] = torch.cat(shift_gt, dim=1)
        if self.use_l1:
            dt['l1'] = ori_reg_dt.permute(1, 0, 2).reshape(4, -1)[:, pos_mask]
            gt['l1'] = torch.cat(l1_gt, dim=1)

        return self.loss(dt, gt)

    def inferencing(self, sample):
        dt = self.core(sample['imgs'])

        # restore the predicting bboxes via pre-defined anchors
        dt[:, :4, :] = self.get_shift_bbox(dt[:, :4, :])
        dt[:, 4, :] = self.sigmoid(dt[:, 4, :])
        dummy_cls_shape = (dt.shape[0], 1, dt.shape[2])
        cls = torch.ones(dummy_cls_shape).to(dt.device)
        dt = torch.cat([dt, cls], dim=1)

        dt = torch.permute(dt, (0, 2, 1))

        result_list = self.nms(dt)

        fin_result = []
        for result, id, ori_shape in zip(result_list, sample['ids'], sample['shapes']):
            fin_result.append(Result(result, id, ori_shape, self.input_shape))
        return fin_result

    def get_shift_bbox(self, ori_box:torch.Tensor): # return xywh Bbox
        shift_box = ori_box.clone().to(torch.float32)
        if self.use_anchor:
            anchors = torch.tile(self.anchs.t(), (shift_box.shape[0], 1, 1))
            shift_box[:, 2:] = anchors[:, 2:] * torch.exp(ori_box[:, 2:4].clamp(max=25))
            shift_box[:, :2] = anchors[:, :2] + ori_box[:, :2] * anchors[:, 2:]
        else:
            anchor_points = torch.tile(self.anchs.t(), (shift_box.shape[0], 1, 1))
            stride = torch.tile(self.stride.t(), (shift_box.shape[0], 1, 1))
            shift_box[:, 2:] = torch.exp(ori_box[:, 2:].clamp(max=25)) * stride
            shift_box[:, :2] = anchor_points + ori_box[:, :2] * stride
        return shift_box

    def get_l1_target(self, gt_pos_mask_generate, pos_mask):
        l1_gt_ib = gt_pos_mask_generate
        if self.use_anchor:
            l1_gt_ib[:, 2:] = torch.log(l1_gt_ib[:, 2:] / self.anchs[pos_mask, 2:])
            l1_gt_ib[:, :2] = (l1_gt_ib[:, :2] - self.anchs[pos_mask, :2]) / self.anchs[pos_mask, 2:]
        else:
            l1_gt_ib[:, 2:] = torch.log(l1_gt_ib[:, 2:] / self.stride[pos_mask] + 1e-8)
            l1_gt_ib[:, :2] = (l1_gt_ib[:, :2] - self.anchs[pos_mask]) / self.stride[pos_mask]
        return l1_gt_ib.t()