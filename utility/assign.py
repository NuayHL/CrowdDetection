# this .py is for the assignment methods
import torch
import torch.nn.functional as F

from utility.iou import IOU
from utility.anchors import Anchor, generateAnchors

class AnchorAssign():
    def __init__(self, config, device):
        self.cfg = config
        self.assignType = config.model.assignment_type.lower()
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

    def assign(self, gt, dt=None):
        '''
        using batch_sized data input
        :param gt:aka:"anns":List lenth B, each with np.float32 ann}
        :param dt:aka:"detections": tensor B x Num x (4+1+classes)}
        :return:the same sture of self.anchs, but filled
                with value indicates the assignment of the anchor
        '''
        if self.assignType == "default":
            if self.using_ignored_input:
                return self._retinaAssign_using_ignored(gt)
            else:
                return self._retinaAssign(gt)
        elif self.assignType == 'simota':
            assert dt != None, 'You are using SimOTA, please input dt'
            return self._simOTA(gt, dt)
        else:
            raise NotImplementedError("Unknown assignType: %s"%self.assignType)

    def _retinaAssign(self,gt):
        output_size = (len(gt),self.anchs.shape[0])
        assign_result = torch.zeros(output_size)
        assign_result = assign_result.to(self.device)
        for ib in range(len(gt)):
            imgAnn = gt[ib][:,:4]
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
            iou_max_value = torch.where(iou_max_value >= self.threshold_iou, (iou_max_idx + 2.0).float(),iou_max_value)
            iou_max_value = torch.where(iou_max_value < self.threshold_iou-0.1, 1.0, iou_max_value)
            iou_max_value = torch.where(iou_max_value < self.threshold_iou, 0., iou_max_value)

            # Assign at least one anchor to the gt
            iou_max_value[iou_max_idx_anns] = torch.arange(imgAnn.shape[0]).double().to(self.device) + 2
            assign_result[ib] = iou_max_value-1

        return assign_result, gt

    def _retinaAssign_using_ignored(self,gt):
        ''':return: assign result, real gt(exclude ignored ones), all already to tensor'''
        # initialize assign result
        output_size = (len(gt),self.anchs.shape[0])
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
            iou_max_value = torch.where(iou_max_value >= self.threshold_iou, (iou_max_idx + 2.0).float(),iou_max_value)
            iou_max_value = torch.where(iou_max_value < self.threshold_iou-0.1, 1.0, iou_max_value.double())
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

            assign_result[ib] = iou_max_value-1

        return assign_result, real_gt

    def _simOTA(self, gt, dt):
        pass


class AnchFreeAssign():
    def __init__(self):
        pass
    def assign(self, gt):
        pass


# This code is based on https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py
class SimOTA():
    def __init__(self,config, device):
        self.config = config
        self.device = device
        anchorGen = Anchor(config)
        self.iou = IOU(ioutype=config.model.assignment_iou_type, gt_type='xywh', dt_type='xywh')
        self.anch = torch.from_numpy(anchorGen.gen_points(singleBatch=True)).to(device)
        self.stride = torch.from_numpy(anchorGen.gen_stride(singleBatch=True)).to(device)
        self.num_classes = config.data.numofclasses
        self.num_anch = len(self.anch)

    def __call__(self, gt, shift_dt):
        for ib in range(len(gt)):
            dt_ib = shift_dt[ib]
            gt_ib = torch.from_numpy(gt[ib]).to(self.device)
            in_box_mask_ib, matched_anchor_gt_mask_ib = self.get_in_boxes_info(gt_ib,
                                                                       self.anch,
                                                                       self.stride)
            shift_Bbox_pre_ib_ = shift_dt[ib, :4, in_box_mask_ib]
            dt_obj_ib_ = shift_dt[ib, 4, in_box_mask_ib]
            dt_cls_ib_ = shift_dt[ib, 5:, in_box_mask_ib]\

            num_in_gt_anch_ib = shift_Bbox_pre_ib_.shape[1]
            num_gt_ib = len(gt_ib)

            iou_gt_dt_pre_ib = self.iou(gt_ib, shift_Bbox_pre_ib_)
            iou_loss_ib = - torch.log(iou_gt_dt_pre_ib + 1e-8)

            gt_cls_ib = F.one_hot(gt_ib[:,4].to(torch.int64), self.num_classes)\
                .to(torch.float32).unsqueeze(1).repeat(1, num_in_gt_anch_ib, 1)

            with torch.cuda.amp.autocast(enabled=False):
                dt_cls_ib_ = (dt_cls_ib_.float().unsqueeze(0).repeat(num_gt_ib,1,1).sigmoid_()
                                 * dt_obj_ib_.float().unsqueeze(0).repeat(num_gt_ib,1,1).sigmoid_())
                cls_loss_ib = F.binary_cross_entropy(dt_cls_ib_.sqrt_(), gt_cls_ib, reduction='none').sum(-1)

            cost_ib = (cls_loss_ib + 3.0 * iou_loss_ib + 100000.0 * (~matched_anchor_gt_mask_ib))

            return (self.dynamic_k_matching(cost_ib, iou_gt_dt_pre_ib, gt_ib[:,4], num_gt_ib, in_box_mask_ib),
                    in_box_mask_ib)

    @staticmethod
    def get_in_boxes_info(gt_ib, anchor, stride):
        """
        :param gt_ib: [num_gt, 4]
        :param anchor: [num_anchor, 2]  2:x, y
        :param stride: [num_anchor]
        :return: mask_in_Bbox, mask_in_center&Bbox
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
    def dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
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
        if (anchor_matching_gt > 1).sum() > 0: # deal with the ambigous anchs
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_posi_anch = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        # fix: gt_class no need here
        # fix: num_posi_anch no need here
        cls_weight = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_posi_anch, gt_matched_classes, cls_weight, matched_gt_inds