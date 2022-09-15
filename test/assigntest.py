import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, '../odcore'))
import torch
import torch.nn as nn
from torch import tensor as t
import numpy as np
from odcore.utils.visualization import assign_hot_map
from odcore.data.dataset import CocoDataset
from modelzoo.build_models import BuildModel
from config import get_default_cfg
from utility.assign import AnchorAssign, SimOTA
from utility.anchors import Anchor

device = 1

cfg = get_default_cfg()
cfg.merge_from_files('cfgs/yolov3')

dataset = CocoDataset('CrowdHuman/annotation_train_coco_style.json','CrowdHuman/Images_train',cfg.data, 'train')
sample = dataset[100]
img = sample['img']
gt = sample['anns']

samples = CocoDataset.OD_default_collater([sample])

builder = BuildModel(cfg)
model = builder.build()

assigner = SimOTA(cfg, device)
anchorgen = Anchor(cfg)

model.set(None, device)
model = model.to(device)
samples['imgs'] = samples['imgs'].to(device).float() / 255

para = torch.load('YOLOv3_640.pth')
model.load_state_dict(para['model'])
dt = model.core(samples['imgs'])
anchs = anchorgen.gen_Bbox(singleBatch=True)
anchor_gpu = torch.from_numpy(anchs).to(device)

def get_shift_bbox(ori_box: torch.Tensor):  # return xywh Bbox
    shift_box = ori_box.clone().to(torch.float32)
    anchors = torch.tile(anchor_gpu.t(), (shift_box.shape[0], 1, 1))
    shift_box[:, 2:] = anchors[:, 2:] * torch.exp(ori_box[:, 2:4].clamp(max=25))
    shift_box[:, :2] = anchors[:, :2] + ori_box[:, :2] * anchors[:, 2:]
    return shift_box

dt[:, :4, :] = get_shift_bbox(dt[:, :4, :])

(num_pos, matched_classes, cls_weight, matched_gt), mask = assigner(samples['annss'], dt)

mask = mask.cpu()

assign_hot_map(anchs, mask, (640,640), img, gt)


