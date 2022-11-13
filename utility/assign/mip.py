import torch
import torch.nn.functional as F

from utility.iou import IOU
from utility.anchors import Anchor, generateAnchors
from utility.assign.build import AssignRegister

@AssignRegister.register
@AssignRegister.register('mip')
class MIP():
    def __init__(self, config, device):
        self.config = config
        self.iou = IOU(ioutype=config.model.assignment_iou_type, gt_type='xywh')
        self.threshold_iou = config.model.assignment_iou_threshold
        self.using_ignored_input = config.data.ignored_input
        self.device = device
        self.mip_k = int(self.config.model.assignment_extra[0]['k'])
        genAchor = Anchor(config)

        # used for mip formate transfer
        anchors_per_grid = genAchor.get_anchors_per_grid()
        num_in_each_level = genAchor.get_num_in_each_level()
        self.referencing_table = []
        for level_num in num_in_each_level:
            self.referencing_table += [int(level_num/anchors_per_grid)] * anchors_per_grid

        self.anchs = torch.from_numpy(genAchor.gen_Bbox(singleBatch=True, act_mip_mode_if_use=False)).float().to(device)
        # change anchor format from xywh to x1y1x2y2
        self.anchs[:, 0] = self.anchs[:, 0] - 0.5 * self.anchs[:, 2]
        self.anchs[:, 1] = self.anchs[:, 1] - 0.5 * self.anchs[:, 3]
        self.anchs[:, 2] = self.anchs[:, 0] + self.anchs[:, 2]
        self.anchs[:, 3] = self.anchs[:, 1] + self.anchs[:, 3]
        self.anchs_len = self.anchs.shape[0]
        self.zero = torch.tensor(0).to(self.device)

    def assign(self, gt):
        output_size = (len(gt), self.anchs.shape[0] * self.mip_k)
        assign_result = torch.zeros(output_size)
        weight = []
        fin_gt = []
        assign_result = assign_result.to(self.device)
        for ib in range(len(gt)):
            gt_ib = torch.from_numpy(gt[ib]).float().to(self.device)
            fin_gt.append(gt_ib)
            assign_result_ib = torch.zeros(self.anchs_len * self.mip_k, device=self.device)
            if len(gt_ib) == 0:
                assign_result[ib] = assign_result_ib
                weight.append(torch.zeros(1, device=self.device))
                continue
            iou_matrix = self.iou(self.anchs, gt_ib[:, :4])
            pad = self.mip_k - iou_matrix.shape[1]
            if pad > 0:
                iou_matrix = F.pad(iou_matrix, (0, 2), value=0.0)
            real_iou_matrix = iou_matrix.clone()
            _, iou_max_idx_anns = torch.max(iou_matrix, dim=0)
            iou_matrix[iou_max_idx_anns, torch.arange(len(gt_ib)).long()] = 0.95

            iou_max_value, iou_max_idx = torch.topk(iou_matrix, k=self.mip_k, dim=1)
            real_iou_max_value = real_iou_matrix[torch.arange(self.anchs_len).long(), iou_max_idx.t()].t()

            iou_max_value = self.mip_shape_transfer(iou_max_value)
            real_iou_max_value = self.mip_shape_transfer(real_iou_max_value)
            test = (iou_max_value - real_iou_max_value).sum()
            iou_max_idx = self.mip_shape_transfer(iou_max_idx)

            pos_mask = torch.gt(iou_max_value, self.threshold_iou)
            # negative: 0
            # positive: index+1
            assign_result_ib[pos_mask] = (iou_max_idx[pos_mask] + 1.0).float()
            weight_ib = real_iou_max_value[pos_mask].float()
            assign_result[ib] = assign_result_ib
            weight.append(weight_ib)

        return assign_result, fin_gt, weight

    def mip_shape_transfer(self, assign_value_like):
        output_assign = torch.zeros(assign_value_like.shape[0]*self.mip_k,
                                    dtype=assign_value_like.dtype,
                                    device=self.device)
        index_fin = 0
        index_input = 0
        for num in self.referencing_table:
            for i in range(self.mip_k):
                output_assign[index_fin:index_fin+num] = assign_value_like[index_input:index_input+num, i]
                index_fin += num
            index_input += num
        return output_assign
