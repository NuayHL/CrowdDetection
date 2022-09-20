import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, 'odcore'))
from odcore.utils.visualization import draw_scheduler
from config import get_default_cfg
from odcore.utils.lr_schedular import LFScheduler

cfg = get_default_cfg()
cfg.merge_from_files('cfgs/yolox')

builder = LFScheduler(cfg)
lf = builder.get_lr_fun()
draw_scheduler(lf, 400)