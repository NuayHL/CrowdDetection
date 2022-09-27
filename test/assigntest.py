import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, '../odcore'))
import torch
import torch.nn as nn
from torch import tensor as t
import numpy as np
import matplotlib.pyplot as plt
from odcore.utils.visualization import assign_hot_map, stack_img, generate_hot_bar
from odcore.data.dataset import CocoDataset
from modelzoo.build_models import BuildModel
from config import get_default_cfg
from utility.assign import AnchorAssign, SimOTA
from utility.anchors import Anchor, result_parse

device = 0

cfg = get_default_cfg()
cfg.merge_from_files('../cfgs/yolox_free_s2')

dataset = CocoDataset('../CrowdHuman/annotation_train_coco_style.json','../CrowdHuman/Images_train',cfg.data, 'val')
sample = dataset[200]
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

para = torch.load('../running_log/YOLOX_640_OTA_free/best_epoch.pth')
model.load_state_dict(para['model'])
dt = model.core(samples['imgs'])
if cfg.model.use_anchor:
    anchs = anchorgen.gen_Bbox(singleBatch=True)
    anchs = torch.from_numpy(anchs).to(device)
else:
    anchs = torch.from_numpy(anchorgen.gen_points(singleBatch=True)).float().to(device)
    single_stride = torch.from_numpy(anchorgen.gen_stride(singleBatch=True)).float().to(device).unsqueeze(1)
    stride = torch.cat([single_stride, single_stride], dim=1)

def get_shift_bbox(ori_box: torch.Tensor):  # return xywh Bbox
    shift_box = ori_box.clone().to(torch.float32)
    if cfg.model.use_anchor:
        anchors = torch.tile(anchs.t(), (shift_box.shape[0], 1, 1))
        shift_box[:, 2:] = anchors[:, 2:] * torch.exp(ori_box[:, 2:4].clamp(max=25))
        shift_box[:, :2] = anchors[:, :2] + ori_box[:, :2] * anchors[:, 2:]
    else:
        anchor_points = torch.tile(anchs.t(), (shift_box.shape[0], 1, 1))
        strides = torch.tile(stride.t(), (shift_box.shape[0], 1, 1))
        shift_box[:, 2:] = torch.exp(ori_box[:, 2:].clamp(max=25)) * strides
        shift_box[:, :2] = anchor_points + ori_box[:, :2] * strides
    return shift_box

dt[:, :4, :] = get_shift_bbox(dt[:, :4, :])

mask_ota, _, _ = assigner_ota.assign(samples['annss'], dt)

mask_ota = mask_ota[0].cpu()

if cfg.model.use_anchor:
    anchs = anchs.cpu().numpy()
else:
    anchs = torch.cat([anchs, stride], dim = 1)
    anchs = anchs.cpu().numpy()

clampled_mask = mask_ota.clamp(0,1)
print(clampled_mask.sum())

clampled_mask = clampled_mask.unsqueeze(dim=1)

level_mask = result_parse(cfg, clampled_mask, restore_size=True)

sum_result = []

fig, ax = plt.subplots(3,4)
for id, level in enumerate(level_mask):
    for il, fm in enumerate(level):
        fm = fm.numpy()
        sum_result.append(fm)
        ax[id][il].imshow(fm)
        ax[id][il].axis('off')
plt.show()

fpnlevels = len(cfg.model.fpnlevels)
anchor_per_grid = len(cfg.model.anchor_ratios) * len(cfg.model.anchor_scales) if cfg.model.use_anchor else 1

sum_result = stack_img(sum_result, (fpnlevels, anchor_per_grid))

bar = generate_hot_bar(1.0, 0.0, sum_result.shape[0])
sum_result = np.concatenate([sum_result, bar], axis=1)
fig, ax = plt.subplots()
ax.imshow(sum_result)
ax.axis('off')
plt.show()

assign_hot_map(anchs, mask_ota, (640,640), img, gt)

# mask_norm, _ = assigner_norm.assign(samples['annss'])
# mask_norm = mask_norm[0]
# assign_hot_map(anchs, mask_norm, (640,640), img, gt)



