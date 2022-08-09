import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
import torch
import torch.nn as nn
from torch import tensor as t
import numpy as np
from odcore.utils.visualization import draw_loss, assign_hot_map
from odcore.data.dataset import CocoDataset
from config import get_default_cfg
from utility.assign import AnchorAssign
from utility.anchors import generateAnchors

cfg = get_default_cfg()
cfg.merge_from_file('letterbox.yaml')

dataset = CocoDataset('CrowdHuman/annotation_train_coco_style.json','CrowdHuman/Images_train',cfg.data, 'train')

# for i, sample in enumerate(dataset):
#     if sample['id'] == '282555,1e085000ad509d2c':
#         id = i
#         break

sample = dataset[1000]

img = sample['img']
gt = sample['anns']

anchors = generateAnchors(cfg, singleBatch=True)
assign_method = AnchorAssign(cfg, 'cpu')

assign ,gt= assign_method.assign([gt])
assign = assign[0, :]
gt = gt[0]
gt = gt.numpy()
assign_hot_map(anchors, assign, (640,640), img, gt)



assign ,gt= assign_method.assign([gt])
assign = assign[0, :]
gt = gt[0]
gt = gt.numpy()
assign_hot_map(anchors, assign, (640,640), img, gt)

assign ,gt= assign_method.assign([gt])
assign = assign[0, :]
gt = gt[0]
gt = gt.numpy()
assign_hot_map(anchors, assign, (640,640), img, gt)
