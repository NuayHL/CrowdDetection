import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, 'odcore'))
import torch
import torch.nn as nn
from torch import tensor as t
import numpy as np
from odcore.utils.visualization import show_bbox, LossLog, draw_coco_eval
from odcore.data.dataset import check_anno_bbox
from odcore.data.dataset import CocoDataset
from modelzoo.build_models import BuildModel
from config import get_default_cfg
from odcore.utils.exp import Exp
from utility.anchors import generateAnchors, anchors_parse, Anchor

draw_coco_eval("running_log/YOLOX_ori/YOLOX_ori_val.log")

# cfg = get_default_cfg()
# cfg.merge_from_files('cfgs/anchortest.yaml')
#
# print(hasattr(cfg, 'training'))

#merge_from_files(cfg, 'cfgs/yolox')

# show_bbox(sample['img'], sample['anns'])

# check_anno_bbox('CrowdHuman/annotation_val_coco_style.json')
# builder = BuildModel(cfg)
# model = builder.build()

# dataset = CocoDataset('CrowdHuman/annotation_val_coco_style_checked.json', 'CrowdHuman/Images_val', cfg.data, 'val')
# sample = dataset[299]
# show_bbox(sample['img'], sample['anns'])
# labels = sample['anns']
# print(labels.shape)
#
# anchor = Anchor(config=cfg)
# anchor_points = anchor.gen_points(singleBatch=True)
#
# x_anchor_center = anchor_points[:,0]
# y_anchor_center = anchor_points[:,1]
#
# print(anchor.gen_stride())
from odcore.utils.misc import xywh_x1y1x2y2


