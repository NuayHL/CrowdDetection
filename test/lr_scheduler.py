import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, '../odcore'))
os.chdir(os.path.join(path, '..'))
from odcore.utils.visualization import draw_scheduler
from config import get_default_cfg
from odcore.utils.lr_schedular import LFScheduler

import matplotlib
matplotlib.use('TkAgg')

cfg = get_default_cfg()
cfg.merge_from_files('cfgs/yolox_pd_csp_poly')

builder = LFScheduler(cfg)
lf = builder.get_lr_fun()
draw_scheduler(lf, 400)