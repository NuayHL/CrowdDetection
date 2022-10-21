# this .py is for the assignment methods
import torch
import torch.nn.functional as F

from utility.iou import IOU
from utility.anchors import Anchor
from utility.assign.build import AssignRegister


# This code is modified from https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py
@AssignRegister.register
@AssignRegister.register('simOTA')
def simota(config, device):
    if config.data.ignored_input:
        return SimOTA_UsingIgnored(config, device)
    else:
        return SimOTA(config, device)


class SimOTA:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        anchorGen = Anchor(config)
        self.iou = IOU(ioutype=config.model.assignment_iou_type, gt_type='xywh', dt_type='xywh')
        self.using_anchor = config.model.use_anchor
        if self.using_anchor:
            self.anchs = torch.from_numpy(anchorGen.gen_Bbox(singleBatch=True)).to(device)
        else:
            self.anchs = torch.from_numpy(anchorGen.gen_points(singleBatch=True)).to(device)
        self.stride = torch.from_numpy(anchorGen.gen_stride(singleBatch=True)).to(device)
        self.num_classes = config.data.numofclasses
        self.num_anch = len(self.anchs)

    def assign(self, gt, shift_dt):
        output_size = (len(gt), self.num_anch)
        assign_result = torch.zeros(output_size).to(self.device)
        cls_weights = []
        fin_gt = []

        for ib in range(len(gt)):
            dt_ib = shift_dt[ib].t()
            gt_ib = torch.from_numpy(gt[ib]).to(self.device)

            if len(gt_ib) == 0:  # deal with blank image
                assign_result[ib] = torch.zeros(self.num_anch)
                cls_weights.append(0)
                fin_gt.append(gt_ib)
                continue

            in_box_mask_ib, matched_anchor_gt_mask_ib = self.get_in_boxes_info(gt_ib,
                                                                               self.anchs,
                                                                               self.stride)
            shift_Bbox_pre_ib_ = dt_ib[in_box_mask_ib, :4]
            dt_obj_ib_ = dt_ib[in_box_mask_ib, 4:5]
            dt_cls_ib_ = dt_ib[in_box_mask_ib, 5:]

            num_in_gt_anch_ib = shift_Bbox_pre_ib_.shape[0]
            num_gt_ib = len(gt_ib)

            iou_gt_dt_pre_ib = self.iou(gt_ib, shift_Bbox_pre_ib_)
            iou_loss_ib = - torch.log(iou_gt_dt_pre_ib + 1e-5)

            gt_cls_ib = gt_ib[:, 4].to(torch.int64)

            gt_cls_ib = F.one_hot(gt_cls_ib, self.num_classes) \
                .to(torch.float32).unsqueeze(1).repeat(1, num_in_gt_anch_ib, 1)

            with torch.cuda.amp.autocast(enabled=False):
                dt_cls_ib_ = (dt_cls_ib_.float().unsqueeze(0).repeat(num_gt_ib, 1, 1).sigmoid_()
                              * dt_obj_ib_.float().unsqueeze(0).repeat(num_gt_ib, 1, 1).sigmoid_())
                cls_loss_ib = F.binary_cross_entropy(dt_cls_ib_.sqrt_(), gt_cls_ib, reduction='none').sum(-1)

            cost_ib = (cls_loss_ib + 3.0 * iou_loss_ib + 100000.0 * (~matched_anchor_gt_mask_ib))

            matched_id_ib, cls_ib_weight = self.dynamic_k_matching(cost_ib, iou_gt_dt_pre_ib, num_gt_ib, in_box_mask_ib)

            # negative: 0
            # positive: index+1
            assignment = in_box_mask_ib.float()
            assignment[in_box_mask_ib.clone()] = matched_id_ib.float() + 1

            assign_result[ib] = assignment
            cls_weights.append(cls_ib_weight)
            fin_gt.append(gt_ib)

        return assign_result, fin_gt, cls_weights

    @staticmethod
    def get_in_boxes_info(gt_ib, anchor, stride):
        """
        :param gt_ib: [num_gt, 4]
        :param anchor: [num_anchor, 2]  2:x, y
        :param stride: [num_anchor]
        :return: mask_for_anchors, mask_in_[num_gt X masked_anchors]
        """
        total_num_anchors = len(anchor)
        num_gt = len(gt_ib)

        anchor = anchor.unsqueeze(0).repeat(num_gt, 1, 1)
        gt_l = (gt_ib[:, 0] - gt_ib[:, 2] * 0.5).unsqueeze(1).repeat(1, total_num_anchors)
        gt_r = (gt_ib[:, 0] + gt_ib[:, 2] * 0.5).unsqueeze(1).repeat(1, total_num_anchors)
        gt_t = (gt_ib[:, 1] - gt_ib[:, 3] * 0.5).unsqueeze(1).repeat(1, total_num_anchors)
        gt_b = (gt_ib[:, 1] + gt_ib[:, 3] * 0.5).unsqueeze(1).repeat(1, total_num_anchors)

        b_l = anchor[:, :, 0] - gt_l
        b_r = gt_r - anchor[:, :, 0]
        b_t = anchor[:, :, 1] - gt_t
        b_b = gt_b - anchor[:, :, 1]
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = gt_ib[:, 0].unsqueeze(1).repeat(1, total_num_anchors) - center_radius * stride
        gt_bboxes_per_image_r = gt_ib[:, 0].unsqueeze(1).repeat(1, total_num_anchors) + center_radius * stride
        gt_bboxes_per_image_t = gt_ib[:, 1].unsqueeze(1).repeat(1, total_num_anchors) - center_radius * stride
        gt_bboxes_per_image_b = gt_ib[:, 1].unsqueeze(1).repeat(1, total_num_anchors) + center_radius * stride

        c_l = anchor[:, :, 0] - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - anchor[:, :, 0]
        c_t = anchor[:, :, 1] - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - anchor[:, :, 1]
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    @staticmethod
    def dynamic_k_matching(cost, pair_wise_ious, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:  # deal with the ambigous anchs
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0  # the final assigned ones among the in anchors

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

        cls_weight = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return matched_gt_inds, cls_weight


@AssignRegister.register
@AssignRegister.register('pdsimota')
class SimOTA_PD(SimOTA):
    def assign(self, gt, shift_dt):
        output_size = (len(gt), self.num_anch)
        assign_result = torch.zeros(output_size).to(self.device)
        obj_weights = []
        fin_gt = []

        for ib in range(len(gt)):
            dt_ib = shift_dt[ib].t()
            gt_ib = torch.from_numpy(gt[ib]).to(self.device)

            if len(gt_ib) == 0:  # deal with blank image
                assign_result[ib] = torch.zeros(self.num_anch)
                obj_weights.append(0)
                fin_gt.append(gt_ib)
                continue

            in_box_mask_ib, matched_anchor_gt_mask_ib = self.get_in_boxes_info(gt_ib,
                                                                               self.anchs,
                                                                               self.stride)
            shift_Bbox_pre_ib_ = dt_ib[in_box_mask_ib, :4]
            dt_obj_ib_ = dt_ib[in_box_mask_ib, 4:5]

            num_in_gt_anch_ib = shift_Bbox_pre_ib_.shape[0]
            num_gt_ib = len(gt_ib)

            iou_gt_dt_pre_ib = self.iou(gt_ib, shift_Bbox_pre_ib_)
            iou_loss_ib = - torch.log(iou_gt_dt_pre_ib + 1e-5)

            gt_cls_ib = gt_ib[:, 4].to(torch.int64)

            gt_cls_ib = F.one_hot(gt_cls_ib, self.num_classes) \
                .to(torch.float32).unsqueeze(1).repeat(1, num_in_gt_anch_ib, 1)

            with torch.cuda.amp.autocast(enabled=False):
                dt_cls_ib_ = dt_obj_ib_.float().unsqueeze(0).repeat(num_gt_ib, 1, 1).sigmoid_()
                cls_loss_ib = F.binary_cross_entropy(dt_cls_ib_.sqrt_(), gt_cls_ib, reduction='none').sum(-1)

            cost_ib = (cls_loss_ib + 3.0 * iou_loss_ib + 100000.0 * (~matched_anchor_gt_mask_ib))

            matched_id_ib, cls_ib_weight = self.dynamic_k_matching(cost_ib, iou_gt_dt_pre_ib, num_gt_ib, in_box_mask_ib)

            # negative: 0
            # positive: index+1
            assignment = in_box_mask_ib.float()
            assignment[in_box_mask_ib.clone()] = matched_id_ib.float() + 1

            assign_result[ib] = assignment
            obj_weights.append(cls_ib_weight)
            fin_gt.append(gt_ib)

        return assign_result, fin_gt, obj_weights


class SimOTA_UsingIgnored(SimOTA):
    def __init__(self, config, device):
        super(SimOTA_UsingIgnored, self).__init__(config, device)
        self.half_iou = IOU(ioutype='half_iou', gt_type='xywh', dt_type='xywh')

    def assign(self, gt, shift_dt):
        output_size = (len(gt), self.num_anch)
        assign_result = torch.zeros(output_size).to(self.device)
        cls_weights = []
        fin_gt = []

        for ib in range(len(gt)):
            dt_ib = shift_dt[ib].t()
            gt_raw_ib = torch.from_numpy(gt[ib]).to(self.device).float()
            ignored = torch.eq(gt_raw_ib[:, 4].int(), -1)  # find ignored input
            gt_ib = gt_raw_ib[~ignored]
            gt_ignored_ib = gt_raw_ib[ignored]

            if len(gt_ib) == 0:  # deal with blank image
                assign_result[ib] = torch.zeros(self.num_anch)
                cls_weights.append(0)
                fin_gt.append(gt_ib)
                continue

            in_box_mask_ib, matched_anchor_gt_mask_ib = self.get_in_boxes_info(gt_ib,
                                                                               self.anchs,
                                                                               self.stride)
            shift_Bbox_pre_ib_ = dt_ib[in_box_mask_ib, :4]
            dt_obj_ib_ = dt_ib[in_box_mask_ib, 4:5]
            dt_cls_ib_ = dt_ib[in_box_mask_ib, 5:]

            num_in_gt_anch_ib = shift_Bbox_pre_ib_.shape[0]
            num_gt_ib = len(gt_ib)

            iou_gt_dt_pre_ib = self.iou(gt_ib, shift_Bbox_pre_ib_)
            iou_loss_ib = - torch.log(iou_gt_dt_pre_ib + 1e-5)

            gt_cls_ib = gt_ib[:, 4].to(torch.int64)

            gt_cls_ib = F.one_hot(gt_cls_ib, self.num_classes) \
                .to(torch.float32).unsqueeze(1).repeat(1, num_in_gt_anch_ib, 1)

            with torch.cuda.amp.autocast(enabled=False):
                dt_cls_ib_ = (dt_cls_ib_.float().unsqueeze(0).repeat(num_gt_ib, 1, 1).sigmoid_()
                              * dt_obj_ib_.float().unsqueeze(0).repeat(num_gt_ib, 1, 1).sigmoid_())
                cls_loss_ib = F.binary_cross_entropy(dt_cls_ib_.sqrt_(), gt_cls_ib, reduction='none').sum(-1)

            cost_ib = (cls_loss_ib + 3.0 * iou_loss_ib + 100000.0 * (~matched_anchor_gt_mask_ib))

            matched_id_ib, cls_ib_weight = self.dynamic_k_matching(cost_ib, iou_gt_dt_pre_ib, num_gt_ib, in_box_mask_ib)

            # negative: 0
            # ignore: -1
            # positive: index+1
            assignment = in_box_mask_ib.float()
            assignment[in_box_mask_ib.clone()] = matched_id_ib.float() + 1

            if gt_ignored_ib.shape[0] > 0:
                ignored_mask_ib = self.exclude_ingorned_proposal(dt_ib, gt_ignored_ib)
                assignment[ignored_mask_ib] = -1
                weight_ignored_mask_ib = ~ignored_mask_ib[in_box_mask_ib]
                cls_ib_weight = cls_ib_weight[weight_ignored_mask_ib]

            assign_result[ib] = assignment
            cls_weights.append(cls_ib_weight)
            fin_gt.append(gt_ib)

        return assign_result, fin_gt, cls_weights

    def exclude_ingorned_proposal(self, shift_box, ignored_gt):
        ignored_weight = self.half_iou(shift_box[:, :4], ignored_gt).sum(dim=1)
        ignored_mask = torch.gt(ignored_weight, 0.7)
        return ignored_mask
