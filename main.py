import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
import torch
import torch.nn as nn
from torch import tensor as t
from odcore.utils.visualization import draw_loss, assign_visualization
from odcore.data.dataset import CocoDataset
from config import get_default_cfg
from utility.assign import AnchorAssign
from utility.anchors import generateAnchors

cfg = get_default_cfg()
cfg.merge_from_file('letterbox.yaml')

dataset = CocoDataset('CrowdHuman/annotation_train_coco_style.json','CrowdHuman/Images_train',cfg, 'train')
sample = dataset[1000]

img = sample['img']
gt = sample['anns']

anchors = generateAnchors(cfg, singleBatch=True)
assign_method = AnchorAssign(cfg, 'cpu')
assign = assign_method.assign(gt)
