import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, 'odcore'))
import torch
import torch.nn as nn
from torch import tensor as t
import numpy as np
from odcore.utils.visualization import show_bbox, LossLog, draw_scheduler
from odcore.data.dataset import check_anno_bbox
from odcore.data.dataset import CocoDataset
from modelzoo.build_models import BuildModel
from config import get_default_cfg
from odcore.config import merge_from_files
from odcore.utils.lr_schedular import LFScheduler

cfg = get_default_cfg()
cfg.merge_from_files('cfgs/yolox')
#merge_from_files(cfg, 'cfgs/yolox')

# dataset = CocoDataset('CrowdHuman/annotation_val_coco_style_checked.json', 'CrowdHuman/Images_val', cfg.data, 'val')
#
# sample = dataset[110]
#
# show_bbox(sample['img'], sample['anns'])

# check_anno_bbox('CrowdHuman/annotation_val_coco_style.json')
# builder = BuildModel(cfg)
# model = builder.build()


builder = LFScheduler(cfg)


lf = builder.get_lr_fun()

draw_scheduler(lf, cfg.training.final_epoch)


