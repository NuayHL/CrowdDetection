import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, '../odcore'))
import torch
import torch.nn as nn
from torch import tensor as t
import numpy as np
from odcore.utils.visualization import show_bbox
from odcore.data.dataset import CocoDataset
from config import get_default_cfg

cfg = get_default_cfg()
cfg.merge_from_file('default_config.yaml')

dataset = CocoDataset('CrowdHuman/annotation_train_coco_style.json','CrowdHuman/Images_train',cfg.data, 'train')

sample = dataset[1998]

show_bbox(sample['img'], sample['anns'])
