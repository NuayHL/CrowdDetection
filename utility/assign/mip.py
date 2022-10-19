import torch

from utility.iou import IOU
from utility.anchors import Anchor, generateAnchors

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
