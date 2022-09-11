import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, 'odcore'))
import torch
import torch.nn as nn
from torch import tensor as t
import numpy as np
from odcore.utils.visualization import show_bbox, LossLog
from odcore.data.dataset import check_anno_bbox
from odcore.data.dataset import CocoDataset
from modelzoo.build_models import BuildModel
from config import get_default_cfg
from odcore.utils.exp import Exp
from utility.anchors import generateAnchors, anchors_parse, Anchor

cfg = get_default_cfg()
cfg.merge_from_files('cfgs/anchortest.yaml')
#
# print(hasattr(cfg, 'training'))

#merge_from_files(cfg, 'cfgs/yolox')

# show_bbox(sample['img'], sample['anns'])

# check_anno_bbox('CrowdHuman/annotation_val_coco_style.json')
# builder = BuildModel(cfg)
# model = builder.build()

dataset = CocoDataset('CrowdHuman/annotation_val_coco_style_checked.json', 'CrowdHuman/Images_val', cfg.data, 'val')
sample = dataset[299]
show_bbox(sample['img'], sample['anns'])
labels = sample['anns']
print(labels.shape)

anchor = Anchor(config=cfg)
anchor_points = anchor.gen_points(singleBatch=True)

x_anchor_center = anchor_points[:,0]
y_anchor_center = anchor_points[:,1]

print(anchor.gen_stride())
from odcore.utils.misc import xywh_x1y1x2y2

def get_in_boxes_info(gt_ib, anchor, stride):
    total_num_anchors = len(anchor)
    num_gt = len(gt_ib)

    anchor = anchor.unsqueeze(0).repeat(num_gt,1,1)
    gt_ib_ = xywh_x1y1x2y2(gt_ib).unsqueeze(1).repeat(1,total_num_anchors,1)

    b_l = anchor[:,:,0] - gt_ib_[:, :, 0]
    b_r = gt_ib_[:, :, 2] - anchor[:,:,0]
    b_t = anchor[:,:,1] - gt_ib_[:, :, 1]
    b_b = gt_ib_[:, :, 3] - anchor[:,:,1]
    bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

    is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
    is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
    # in fixed center

    center_radius = 2.5

    gt_bboxes_per_image_l = gt_ib[:, 0].unsqueeze(1).repeat(1, total_num_anchors) - center_radius * stride
    gt_bboxes_per_image_r = gt_ib[:, 0].unsqueeze(1).repeat(1, total_num_anchors) + center_radius * stride
    gt_bboxes_per_image_t = gt_ib[:, 1].unsqueeze(1).repeat(1, total_num_anchors) - center_radius * stride
    gt_bboxes_per_image_b = gt_ib[:, 1].unsqueeze(1).repeat(1, total_num_anchors) + center_radius * stride

    c_l = anchor[:,:,0] - gt_bboxes_per_image_l
    c_r = gt_bboxes_per_image_r - anchor[:,:,0]
    c_t = anchor[:,:,1] - gt_bboxes_per_image_t
    c_b = gt_bboxes_per_image_b - anchor[:,:,1]
    center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
    is_in_centers = center_deltas.min(dim=-1).values > 0.0
    is_in_centers_all = is_in_centers.sum(dim=0) > 0

    # in boxes and in centers
    is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

    is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
    )
    return is_in_boxes_anchor, is_in_boxes_and_center

