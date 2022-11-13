# this .py is for the assignment methods
import torch
import torch.nn.functional as F

from utility.iou import IOU
from utility.anchors import Anchor
from utility.assign.build import AssignRegister

@AssignRegister.register
@AssignRegister.register('default')
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
            pos_mask = torch.gt(iou_max_value, self.threshold_iou)
            neg_mask = torch.lt(iou_max_value, self.threshold_iou - 0.1)
            assign_result_ib = torch.zeros_like(iou_max_value)
            assign_result_ib[pos_mask] = (iou_max_idx[pos_mask] + 2.0).float()
            assign_result_ib[neg_mask] = 1.0

            # Assign at least one anchor to the gt
            assign_result_ib[iou_max_idx_anns] = torch.arange(imgAnn.shape[0]).float().to(self.device) + 2
            assign_result[ib] = assign_result_ib - 1

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
            _, iou_max_idx_anns = torch.max(iou_matrix, dim=0)
            # negative: 0
            # ignore: -1
            # positive: index+1
            pos_mask = torch.gt(iou_max_value, self.threshold_iou)
            neg_mask = torch.lt(iou_max_value, self.threshold_iou - 0.1)
            assign_result_ib = torch.zeros_like(iou_max_value)
            assign_result_ib[pos_mask] = (iou_max_idx[pos_mask] + 2.0).float()
            assign_result_ib[neg_mask] = 1.0

            # Assign at least one anchor to the gt
            assign_result_ib[iou_max_idx_anns] = torch.arange(imgAnn.shape[0]).float().to(self.device) + 2
            assign_result_ib = assign_result_ib.int()
            # Dealing with ignored area
            if ignoredAnn.shape[0] != 0:
                false_sample_idx = torch.eq(assign_result_ib, 1)
                ignore_iou_matrix = self.iou(self.anchs[false_sample_idx], ignoredAnn)
                false_sample = assign_result_ib[false_sample_idx]
                ignored_max_iou_value, _ = torch.max(ignore_iou_matrix, dim=1)
                # set the iou threshold as 0.5
                ignored_anchor_idx = torch.ge(ignored_max_iou_value, self.threshold_iou)
                false_sample[ignored_anchor_idx] = 0
                assign_result_ib[false_sample_idx] = false_sample

            assign_result[ib] = assign_result_ib - 1

        return assign_result, real_gt, None
