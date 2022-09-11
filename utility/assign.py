# this .py is for the assignment methods
import torch

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
        # change anchor format from xywh to x1y1x2y2
        #self.anchs = torch.from_numpy(generateAnchors(config,singleBatch=True)).float().to(device)
        genAchor = Anchor(config)
        self.anchs = torch.from_numpy(genAchor.gen_Bbox(singleBatch=True)).float().to(device)
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
            imgAnn = torch.from_numpy(imgAnn).double()
            if torch.cuda.is_available():
                imgAnn = imgAnn.to(self.device)

            iou_matrix = self.iou(self.anchs, imgAnn)
            iou_max_value, iou_max_idx = torch.max(iou_matrix, dim=1)
            iou_max_value_anns, iou_max_idx_anns = torch.max(iou_matrix, dim=0)
            # negative: 0
            # ignore: -1
            # positive: index+1
            iou_max_value = torch.where(iou_max_value >= self.threshold_iou, iou_max_idx.double() + 2.0,iou_max_value)
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
            imgAnn = imgAnn[~ignored]
            if imgAnn.shape[0] == 0:
                assign_result[ib] = torch.zeros(self.anchs.shape[0]).to(self.device)
                continue

            iou_matrix = self.iou(self.anchs, imgAnn)
            iou_max_value, iou_max_idx = torch.max(iou_matrix, dim=1)
            iou_max_value_anns, iou_max_idx_anns = torch.max(iou_matrix, dim=0)
            # negative: 0
            # ignore: -1
            # positive: index+1
            iou_max_value = torch.where(iou_max_value >= self.threshold_iou, iou_max_idx + 2.0,iou_max_value)
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

class SimOTA():
    def __init__(self):
        pass

    def __call__(self, grid, gt):
        pass

def get_in_boxes_info(
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
):
    expanded_strides_per_image = expanded_strides[0]
    x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
    y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
    x_centers_per_image = (
        (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
    )  # [n_anchor] -> [n_gt, n_anchor]
    y_centers_per_image = (
        (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
    )

    gt_bboxes_per_image_l = (
        (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
    )
    gt_bboxes_per_image_r = (
        (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
    )
    gt_bboxes_per_image_t = (
        (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
    )
    gt_bboxes_per_image_b = (
        (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
    )

    b_l = x_centers_per_image - gt_bboxes_per_image_l
    b_r = gt_bboxes_per_image_r - x_centers_per_image
    b_t = y_centers_per_image - gt_bboxes_per_image_t
    b_b = gt_bboxes_per_image_b - y_centers_per_image
    bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

    is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
    is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
    # in fixed center

    center_radius = 2.5

    gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
        1, total_num_anchors
    ) - center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
        1, total_num_anchors
    ) + center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
        1, total_num_anchors
    ) - center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
        1, total_num_anchors
    ) + center_radius * expanded_strides_per_image.unsqueeze(0)

    c_l = x_centers_per_image - gt_bboxes_per_image_l
    c_r = gt_bboxes_per_image_r - x_centers_per_image
    c_t = y_centers_per_image - gt_bboxes_per_image_t
    c_b = gt_bboxes_per_image_b - y_centers_per_image
    center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
    is_in_centers = center_deltas.min(dim=-1).values > 0.0
    is_in_centers_all = is_in_centers.sum(dim=0) > 0

    # in boxes and in centers
    is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

    is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
    )
    return is_in_boxes_anchor, is_in_boxes_and_center