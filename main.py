import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, '../odcore'))
import torch
import torch.nn as nn
from torch import tensor as t
import numpy as np
from odcore.utils.visualization import show_bbox, LossLog
from odcore.data.dataset import check_anno_bbox
from odcore.data.dataset import CocoDataset
from config import get_default_cfg

# cfg = get_default_cfg()
# cfg.merge_from_file('test/letterbox.yaml')
#
# dataset = CocoDataset('CrowdHuman/annotation_train_coco_style_checked.json','CrowdHuman/Images_train',cfg.data, 'train')
#
# sample = dataset[100]
#
# show_bbox(sample['img'], sample['anns'])

check_anno_bbox('CrowdHuman/annotation_val_coco_style.json')
