from utility.iou import IOU
from utility.anchors import Anchor
from utility.assign.build import AssignRegister

import warnings
import torch

# The following codes is modified from
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/atss_assigner.py
@AssignRegister.register
@AssignRegister.register('atss')
class ATSS:
    def __init__(self, config, device):
        self.topk = 9
        self.alpha = None
        self.using_ignored_input = config.data.ignored_input
        self.ignored_iou_thresholod = 0.5

        self.config = config
        self.device = device
        anch_gen = Anchor(config)
        self.anchs = torch.from_numpy(anch_gen.gen_Bbox(singleBatch=True)).to(device)
        self.anch_poins = self.anchs[:, :2].clone().detach()
        self.num_in_each_level = anch_gen.get_num_in_each_level()
        self.num_anch = len(self.anchs)
        self.num_classes = config.data.numofclasses
        self.iou_calculator = IOU(dt_type='xywh', gt_type='xywh')

    def assign(self, gt, shift_dt=None):

        message = 'Invalid alpha parameter because cls_scores or ' \
                  'bbox_preds are None. If you want to use the ' \
                  'cost-based ATSSAssigner,  please set cls_scores, ' \
                  'bbox_preds and self.alpha at the same time.'

        INF = 100000000
        output_size = (len(gt), self.num_anch)
        assign_result = torch.zeros(output_size).to(self.device)
        fin_gt = []

        for ib in range(len(gt)):
            gt_raw_ib = torch.from_numpy(gt[ib]).to(self.device)
            ignored = torch.eq(gt_raw_ib[:, 4].int(), -1)  # find ignored input
            gt_ib = gt_raw_ib[~ignored]
            gt_ignored_ib = gt_raw_ib[ignored]
            gt_label_ib = gt_ib[:, 4]

            gt_ib_len = len(gt_ib)

            if gt_ib_len == 0:
                assign_result[ib] = torch.zeros(self.num_anch)
                fin_gt.append(gt_ib)
                continue

            if self.alpha is None:
                overlaps = self.iou_calculator(self.anchs, gt_ib)
                if shift_dt is not None:
                    warnings.warn(message)
            else:
                # Dynamic cost ATSSAssigner in DDOD
                assert shift_dt is not None, message
                dt_ib = shift_dt[ib].t()
                dt_cls_ib = dt_ib[:, 5:]
                cls_cost = torch.sigmoid(dt_cls_ib[:, gt_label_ib])
                overlaps = self.iou_calculator(self.anchs, gt_ib)
                assert cls_cost.shape == overlaps.shape #---------------------------------------------------------------
                overlaps = cls_cost**(1 - self.alpha) * overlaps**self.alpha

            # assign 0 by default
            assignment = overlaps.new_full((self.num_anch,), 0, dtype=torch.long)

            # compute center distance between all bbox and gt
            gt_points = gt_ib[:, :2]

            distances = (self.anch_poins.unsqueeze(dim=1) - gt_points.unsqueeze(dim=0)).pow(2).sum(-1).sqrt()

            if self.using_ignored_input and len(gt_ignored_ib) > 0:
                ignore_overlaps = self.iou_calculator(self.anchs, gt_ignored_ib)
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
                ignore_idxs = ignore_max_overlaps > self.ignored_iou_thresholod
                distances[ignore_idxs, :] = INF
                assignment[ignore_idxs] = -1

            # Selecting candidates based on the center distance
            candidate_idxs = []
            start_idx = 0
            for level, bboxes_per_level in enumerate(self.num_in_each_level):
                # on each pyramid level, for each gt,
                # select k bbox whose center are closest to the gt center
                end_idx = start_idx + bboxes_per_level
                distances_per_level = distances[start_idx:end_idx, :]
                selectable_k = min(self.topk, bboxes_per_level)

                _, topk_idxs_per_level = distances_per_level.topk(
                    selectable_k, dim=0, largest=False)
                candidate_idxs.append(topk_idxs_per_level + start_idx)
                start_idx = end_idx
            candidate_idxs = torch.cat(candidate_idxs, dim=0)

            # get corresponding iou for the these candidates, and compute the
            # mean and std, set mean + std as the iou threshold
            candidate_overlaps = overlaps[candidate_idxs, torch.arange(gt_ib_len)]
            overlaps_mean_per_gt = candidate_overlaps.mean(0)
            overlaps_std_per_gt = candidate_overlaps.std(0)
            overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

            is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

            # limit the positive sample's center in gt
            # candidate_offset = torch.arange(gt_len).to(self.device)
            # candidate_offset *= self.num_anch
            # candidate_offset = candidate_offset.unsqueeze(dim=0).expand(self.topk, -1)
            # candidate_idxs += candidate_offset

            for gt_idx in range(gt_ib_len):
                candidate_idxs[:, gt_idx] += gt_idx * self.num_anch
            ep_bboxes_cx = self.anch_poins[:, 0:1].t().expand(gt_ib_len, self.num_anch).contiguous().view(-1)
            ep_bboxes_cy = self.anch_poins[:, 1:2].t().expand(gt_ib_len, self.num_anch).contiguous().view(-1)
            candidate_idxs = candidate_idxs.view(-1)

            # calculate the left, top, right, bottom distance between positive
            # bbox center and gt side
            l_ = ep_bboxes_cx[candidate_idxs].view(-1, gt_ib_len) - gt_ib[:, 0]
            t_ = ep_bboxes_cy[candidate_idxs].view(-1, gt_ib_len) - gt_ib[:, 1]
            r_ = gt_ib[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, gt_ib_len)
            b_ = gt_ib[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, gt_ib_len)
            is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01

            is_pos = is_pos & is_in_gts

            # if an anchor box is assigned to multiple gts,
            # the one with the highest IoU will be selected.
            overlaps_inf = torch.full_like(overlaps,
                                           -INF).t().contiguous().view(-1)
            index = candidate_idxs.view(-1)[is_pos.view(-1)]
            overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
            overlaps_inf = overlaps_inf.view(gt_ib_len, -1).t()

            max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
            assignment[
                max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

            assign_result[ib] = assignment
            fin_gt.append(gt_ib)

        return assign_result, fin_gt, None
