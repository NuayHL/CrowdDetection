import copy
import os
path = os.getcwd()
os.chdir(os.path.join(path, '..'))
import torch

from config import get_default_cfg
from modelzoo.build_models import BuildModel

cfg = get_default_cfg()
cfg.merge_from_files('cfgs/yolox_ori')

build = BuildModel(cfg)
model = build.build()

state_dict = torch.load('running_log/YOLOX_ori/best_epoch.pth')
dict_model = state_dict['model']
dict_backbone = dict()
for key in dict_model.keys():
    if 'backbone' in key:
        dict_backbone[key] = dict_model[key]

model.load_state_dict(dict_backbone, strict=False)

