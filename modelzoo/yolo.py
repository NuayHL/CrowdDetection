import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp

from modelzoo.basemodel import BaseODModel
from utility.assign import get_assign_method
from utility.anchors import generateAnchors, result_parse, Anchor
from utility.loss import GeneralLoss_fix, updata_loss_dict
from utility.nms import non_max_suppression, NMS, SetNMS
from utility.result import Result

# model.set(args, device)
# model.coco_parse_result(results) results: List of prediction

class YoloX(BaseODModel):
    '''4 + 1 + classes'''
    def __init__(self, config, backbone, neck, head):
        super(YoloX, self).__init__()
        self.config = config
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.sigmoid = nn.Sigmoid()
        self.input_shape = (self.config.data.input_width, self.config.data.input_height)

        self.using_mip = self.config.model.assignment_type.lower() in ['mip', 'MIP']
        self.multi_classes = True if self.config.data.numofclasses > 1 else False
        nms = SetNMS if self.using_mip else NMS
        self.nms = nms(config)

    def core(self, input):
        fms = self.backbone(input)
        rea_fms = self.neck(*fms)
        dt = self.head(*rea_fms)
        return dt

    def set(self, args, device):
        self.device = device
        self.assignment = get_assign_method(self.config, device)
        self.assign_type = self.config.model.assignment_type.lower()
        self.anch_gen = Anchor(self.config)
        self.use_anchor = self.config.model.use_anchor
        if self.use_anchor:
            self.anchs = torch.from_numpy(self.anch_gen.gen_Bbox(singleBatch=True)).float().to(device)
        else:
            assert self.assign_type != "default", "Assign type %s do not support anchor free style"%self.assign_type
            self.anchs = torch.from_numpy(self.anch_gen.gen_points(singleBatch=True)).float().to(device)
            single_stride = torch.from_numpy(self.anch_gen.gen_stride(singleBatch=True)).float().to(device).unsqueeze(1)
            self.stride = torch.cat([single_stride, single_stride], dim=1)
        self.num_of_proposal = self.anchs.shape[0]
        # assert self.config.data.ignored_input is True, "Please set the config.data.ignored_input as True"

        self.use_l1 = True if 'l1' in self.config.loss.reg_type else False

        assert True in ['iou' in loss for loss in self.config.loss.reg_type], 'Please adding iou-type loss'
        self.loss = GeneralLoss_fix(self.config, device, reduction='mean')

    def training_loss(self, sample):
        dt = self.core(sample['imgs']) #[B, 4+1+cls, n_anchors_all]
        num_of_class = dt.shape[1]-5
        cls_dt = dt[:, 5:, :].clone()  #[B, cls, n_anchors_all]
        obj_dt = dt[:, 4, :].clone()   #[B, n_anchors_all]
        ori_reg_dt = dt[:, :4, :]      #[B, 4, n_anchors_all]
        shift_dt = self.get_shift_bbox(ori_reg_dt)

        # with amp.autocast(enabled=False):
        with torch.no_grad():
            if self.assign_type == 'simota':
                assign_result, gt, weight = self.assignment.assign(sample['annss'],
                                                                   torch.cat([shift_dt, obj_dt.unsqueeze(1), cls_dt], dim=1))
            else:
                assign_result, gt, weight = self.assignment.assign(sample['annss'])

        pos_mask = []
        effective_mask = []

        cls_gt = []
        obj_gt = []
        shift_gt = []
        if self.use_l1:
            l1_gt = []

        for ib in range(len(gt)):
            assign_result_ib, gt_ib = assign_result[ib], gt[ib]
            pos_mask_ib = torch.gt(assign_result_ib, 0.5)
            pos_mask.append(pos_mask_ib)

            effective_mask_ib = torch.gt(assign_result_ib, -0.5)
            effective_mask.append(effective_mask_ib)

            label_pos_generate = (assign_result_ib[pos_mask_ib] - 1).long()

            shift_gt_ib = gt_ib[label_pos_generate, :4].t()
            shift_gt.append(shift_gt_ib)
            if self.use_l1:
                l1_gt_ib = gt_ib[label_pos_generate, :4]
                l1_gt_ib = self.get_l1_target(l1_gt_ib, pos_mask_ib)
                l1_gt.append(l1_gt_ib)
            cls_gt_ib = torch.zeros((self.config.data.numofclasses, self.num_of_proposal),
                                    dtype=torch.float32).to(self.device)
            cls_gt_ib[gt_ib[label_pos_generate,4].long(), pos_mask_ib] = 1
            cls_gt_ib = cls_gt_ib[:, pos_mask_ib]
            cls_gt.append(cls_gt_ib)

            obj_gt_ib = assign_result_ib.clamp(0, 1)
            if weight:
                obj_gt_ib[pos_mask_ib] *= weight[ib]
            obj_gt_ib = obj_gt_ib[effective_mask_ib]
            obj_gt.append(obj_gt_ib)

        pos_mask = torch.cat(pos_mask, dim=0)
        effective_mask = torch.cat(effective_mask, dim=0)
        dt = {}
        gt = {}
        dt['obj'] = obj_dt.view(-1)[effective_mask]
        gt['obj'] = torch.cat(obj_gt, dim=0)
        dt['cls'] = cls_dt.permute(1, 0, 2).reshape(num_of_class, -1)[:, pos_mask]
        gt['cls'] = torch.cat(cls_gt, dim=1)
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
        dt[:, 4:, :] = self.sigmoid(dt[:, 4:, :])
        dt = torch.permute(dt, (0, 2, 1))

        # result_list = self.nms(dt, class_indepent=self.multi_classes)
        result_list = self.nms(dt, class_indepent=False)

        fin_result = []
        for result, id, ori_shape in zip(result_list, sample['ids'],sample['shapes']):
            fin_result.append(Result(result, id, ori_shape, self.input_shape))
        return fin_result

    def rec_block_output(self, sample):
        if not hasattr(self, 'block_indicator'):
            numofblock = (len(self.config.model.fpnlevels) * self.anch_gen.get_anchors_per_grid())
            self.block_indicator = torch.from_numpy(self.anch_gen.gen_block_indicator()).to(self.device).int()
            self.block_output_rec = torch.zeros(numofblock).to(self.device).long()
            self.block_real_output_rec = torch.zeros(numofblock).to(self.device).long()

        dt = self.core(sample['imgs'])

        block_inidicator = torch.tile(self.block_indicator, (dt.shape[0], 1, 1))

        # restore the predicting bboxes via pre-defined anchors
        dt[:, :4, :] = self.get_shift_bbox(dt[:, :4, :])
        dt[:, 4:, :] = self.sigmoid(dt[:, 4:, :])

        dt = torch.permute(dt, (0, 2, 1))

        output_indicator, real_output_indicator = self.nms.cal_block_output(dt, block_inidicator)

        block_index, block_counts = output_indicator.unique(return_counts=True)
        for block, counts in zip(block_index, block_counts):
            self.block_output_rec[block] += counts

        for real_output in real_output_indicator:
            block_index, block_counts = real_output.unique(return_counts=True)
            for block, counts in zip(block_index, block_counts):
                self.block_real_output_rec[block] += counts

        return None

    def get_stats(self):
        if hasattr(self, 'block_output_rec') and hasattr(self, 'block_real_output_rec'):
            input_count = self.block_output_rec.tolist()
            all_input_count = self.block_output_rec.sum().cpu().item()
            used_count = self.block_real_output_rec.tolist()
            all_used_count = self.block_real_output_rec.sum().cpu().item()
            all_count_ = torch.clamp(self.block_output_rec, min=1)
            effect_ratio = self.block_real_output_rec.float() / all_count_.float()
            effect_ratio_ = effect_ratio.tolist()
            all_effect_ratio = all_used_count / float(all_input_count)
            contribute_ratio = [block_used/float(all_used_count) for block_used in used_count]
            for idx, (ac, uc, er, cr) in enumerate(zip(input_count, used_count, effect_ratio_, contribute_ratio)):
                print('Block %d [output: %10d | used: %8d | effective ratio: %.4f | contribute ratio: %.4f]'
                      % ((idx+1), ac, uc, er, cr))
            print('  All   [output: %10d | used: %8d | effective ratio: %.4f ]'
                  % (all_input_count, all_used_count, all_effect_ratio))
        else:
            pass

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


