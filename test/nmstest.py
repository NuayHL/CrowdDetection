import os
import sys
path = os.getcwd()
os.chdir(os.path.join(path, '..'))
import torch
import torch.nn as nn
from torch import tensor as t
import numpy as np
import matplotlib.pyplot as plt
from odcore.utils.visualization import show_bbox, generate_hot_bar
from odcore.data.dataset import CocoDataset
from modelzoo.build_models import BuildModel
from config import get_default_cfg
from utility.assign import AnchorAssign, SimOTA
from utility.anchors import Anchor, result_parse

import matplotlib
matplotlib.use('TkAgg')

device = 0
image_id = 120
cfg = get_default_cfg()
cfg.merge_from_files('cfgs/test_yolox_mip')

dataset = CocoDataset('CrowdHuman/annotation_train_coco_style.json','CrowdHuman/Images_train',cfg.data, 'val')
sample = dataset[image_id]
img, _ = dataset.get_ori_image(image_id)
gt = sample['anns']

samples = CocoDataset.OD_default_collater([sample])

builder = BuildModel(cfg)
model = builder.build()

model.set(None, device)
model = model.to(device)
samples['imgs'] = samples['imgs'].to(device).float() / 255


# para = torch.load('YOLOX_640_NM.pth')
# model.load_state_dict(para['model'])

model.eval()
dt = model(samples)

dt = dt[0].to_ori_label()

show_bbox(img, dt[:,:4], type='x1y1x2y2')

