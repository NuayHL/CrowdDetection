import os
path = os.getcwd()
os.chdir(os.path.join(path, '..'))
import torch
from modelzoo.build_models import BuildModel
from config import get_default_cfg


cfg = get_default_cfg()
cfg.merge_from_files('cfgs/test_yoloxa_att_neck_v1')

builder = BuildModel(cfg)
model = builder.build()

test = torch.ones((2, 3, cfg.data.input_height, cfg.data.input_width))

result = model.core(test)

print(result.shape)

