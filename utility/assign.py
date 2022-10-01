# this .py is for the assignment methods
import torch
import torch.nn.functional as F

from utility.iou import IOU
from utility.anchors import Anchor, generateAnchors


def get_assign_method(config, device):
    type = config.model.assignment_type.lower()
    if type == 'simota':
        return SimOTA(config, device)
    elif type == 'pdsimota':
        return SimOTA_PD(config, device)
    elif type == 'default':
        return AnchorAssign(config, device)
    else:
        raise NotImplementedError


class AnchorAssign():
    def __init__(self, config, device):
        self.cfg = config
        self.iou = IOU(ioutype=config.model.assignment_iou_type, gt_type='xywh')
        self.threshold_iou = config.model.assignment_iou_threshold
        self.using_ignored_input = config.data.ignored_input
        self.device = device
        genAchor = Anchor(config)
        self.anchs = torch.from_numpy(genAchor.gen_Bbox(singleBatch=True)).float().to(device)
        # change anchor format from xywh to x1y1x2y2
        self.anchs[:, 0] = self.anchs[:, 0] - 0.5 * self.anchs[:, 2]
        self.anchs[:, 1] = self.anchs[:, 1] - 0.5 * self.anchs[:, 3]
        self.anchs[:, 2] = self.anchs[:, 0] + self.anchs[:, 2]
        self.anchs[:, 3] = self.anchs[:, 1] + self.anchs[:, 3]
        self.anchs_len = self.anchs.shape[0]

    def assign(self, gt):
        '''
        using batch_sized data input
        :param gt:aka:"anns":List lenth B, each with np.float32 ann}
        :param dt:aka:"detections": tensor B x Num x (4+1+classes)}
        :return:the same sture of self.anchs, but filled
                with value indicates the assignment of the anchor
        '''
        if self.using_ignored_input:
            return self._retinaAssign_using_ignored(gt)
        else:
            return self._retinaAssign(gt)

    def _retinaAssign(self, gt):
        output_size = (len(gt), self.anchs.shape[0])
        assign_result = torch.zeros(output_size)
        assign_result = assign_result.to(self.device)
        for ib in range(len(gt)):
            imgAnn = gt[ib][:, :4]
            imgAnn = torch.from_numpy(imgAnn).float()
            if torch.cuda.is_available():
                imgAnn = imgAnn.to(self.device)

            iou_matrix = self.iou(self.anchs, imgAnn)
            iou_max_value, iou_max_idx = torch.max(iou_matrix, dim=1)
            iou_max_value_anns, iou_max_idx_anns = torch.max(iou_matrix, dim=0)
            # negative: 0
            # ignore: -1
            # positive: index+1
            print(iou_max_value.dtype)
            iou_max_value = torch.where(iou_max_value >= self.threshold_iou, (iou_max_idx + 2.0).float(), iou_max_value)
            iou_max_value = torch.where(iou_max_value < self.threshold_iou - 0.1, 1.0, iou_max_value)
            iou_max_value = torch.where(iou_max_value < self.threshold_iou, 0., iou_max_value)

            # Assign at least one anchor to the gt
            iou_max_value[iou_max_idx_anns] = torch.arange(imgAnn.shape[0]).double().to(self.device) + 2
            assign_result[ib] = iou_max_value - 1

        return assign_result, gt, None

    def _retinaAssign_using_ignored(self, gt):
        ''':return: assign result, real gt(exclude ignored ones), all already to tensor'''
        # initialize assign result
        output_size = (len(gt), self.anchs.shape[0])
        assign_result = torch.zeros(output_size).to(self.device)

        # prepare return real gt
        real_gt = []

        for ib in range(len(gt)):
            gt_i = torch.from_numpy(gt[ib]).to(self.device)
            ignored = torch.eq(gt_i[:, 4].int(), -1)
            real_gt.append(gt_i[~ignored])
            imgAnn = gt_i[:, :4]
            ignoredAnn = imgAnn[ignored]
            imgAnn = imgAnn[~ignored].float()
            if imgAnn.shape[0] == 0:
                assign_result[ib] = torch.zeros(self.anchs.shape[0]).to(self.device)
                continue

            iou_matrix = self.iou(self.anchs, imgAnn)
            iou_max_value, iou_max_idx = torch.max(iou_matrix, dim=1)
            iou_max_value_anns, iou_max_idx_anns = torch.max(iou_matrix, dim=0)
            # negative: 0
            # ignore: -1
            # positive: index+1
            iou_max_value = torch.where(iou_max_value >= self.threshold_iou, (iou_max_idx + 2.0).float(), iou_max_value)
            iou_max_value = torch.where(iou_max_value < self.threshold_iou - 0.1, 1.0, iou_max_value.double())
            iou_max_value = torch.where(iou_max_value < self.threshold_iou, .0, iou_max_value.double())

            # Assign at least one anchor to the gt
            iou_max_value[iou_max_idx_anns] = torch.arange(imgAnn.shape[0]).double().to(self.device) + 2
            iou_max_value = iou_max_value.int()
            # Dealing with ignored area
            if ignoredAnn.shape[0] != 0:
                false_sample_idx = torch.eq(iou_max_value, 1)
                ignore_iou_matrix = self.iou(self.anchs[false_sample_idx], ignoredAnn)
                false_sample = iou_max_value[false_sample_idx]
                ignored_max_iou_value, _ = torch.max(ignore_iou_matrix, dim=1)
                # set the iou threshold as 0.5
                ignored_anchor_idx = torch.ge(ignored_max_iou_value, self.threshold_iou)
                false_sample[ignored_anchor_idx] = 0
                iou_max_value[false_sample_idx] = false_sample

            assign_result[ib] = iou_max_value - 1

        return assign_result, real_gt, None


# This code is based on https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py
class SimOTA():
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
            # ignore: -1
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
            # ignore: -1
            # positive: index+1
            assignment = in_box_mask_ib.float()
            assignment[in_box_mask_ib.clone()] = matched_id_ib.float() + 1

            assign_result[ib] = assignment
            obj_weights.append(cls_ib_weight)
            fin_gt.append(gt_ib)

        return assign_result, fin_gt, obj_weights


class MIP():
    def __init__(self, config, device):
        self.cfg = config
        self.iou = IOU(ioutype=config.model.assignment_iou_type, gt_type='xywh')
        self.threshold_iou = config.model.assignment_iou_threshold
        self.using_ignored_input = config.data.ignored_input
        self.device = device
        genAchor = Anchor(config)
        self.anchs = torch.from_numpy(genAchor.gen_Bbox(singleBatch=True)).float().to(device)
        # change anchor format from xywh to x1y1x2y2
        self.anchs[:, 0] = self.anchs[:, 0] - 0.5 * self.anchs[:, 2]
        self.anchs[:, 1] = self.anchs[:, 1] - 0.5 * self.anchs[:, 3]
        self.anchs[:, 2] = self.anchs[:, 0] + self.anchs[:, 2]
        self.anchs[:, 3] = self.anchs[:, 1] + self.anchs[:, 3]
        self.anchs_len = self.anchs.shape[0]
        self.zero = torch.tensor(0).to(self.device)

    def assign(self, gt):
        output_size = (len(gt), self.anchs.shape[0], 2)
        assign_result = torch.zeros(output_size)
        weight = torch.zeros(output_size)
        assign_result = assign_result.to(self.device)
        for ib in range(len(gt)):
            imgAnn = gt[ib][:, :4]
            imgAnn = torch.from_numpy(imgAnn).float()
            if torch.cuda.is_available():
                imgAnn = imgAnn.to(self.device)
            iou_matrix = self.iou(self.anchs, imgAnn)
            iou_max_value, iou_max_idx = torch.topk(iou_matrix, k=2, dim=1)
            # negative: 0
            # positive: index+1
            assign_result_ib = torch.where(iou_max_value >= self.threshold_iou,
                                           (iou_max_idx + 1.0).float(),
                                           self.zero)
            weight_ib = torch.where(iou_max_value >= self.threshold_iou,
                                    (iou_max_value).float(),
                                    self.zero)
            assign_result[ib] = assign_result_ib
            weight[ib] = weight_ib
        return assign_result, gt, weight
