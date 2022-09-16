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

dataset = CocoDataset('CrowdHuman/annotation_train_coco_style.json','CrowdHuman/Images_train',cfg.data, 'val')
sample = dataset[233]
img = sample['img']
gt = sample['anns']

samples = CocoDataset.OD_default_collater([sample])

builder = BuildModel(cfg)
model = builder.build()

assigner_ota = SimOTA(cfg, device)
assigner_norm = AnchorAssign(cfg, device)
anchorgen = Anchor(cfg)

model.set(None, device)
model = model.to(device)
samples['imgs'] = samples['imgs'].to(device).float() / 255

# para = torch.load('YOLOv3_640.pth')
# model.load_state_dict(para['model'])
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

mask_ota, cls_weight = assigner_ota.assign(samples['annss'], dt)

mask_ota = mask_ota[0].cpu()
assign_hot_map(anchs, mask_ota, (640,640), img, gt)

# mask_norm, _ = assigner_norm.assign(samples['annss'])
# mask_norm = mask_norm[0]
# assign_hot_map(anchs, mask_norm, (640,640), img, gt)



