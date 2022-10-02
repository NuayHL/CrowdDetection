import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, '../odcore'))
os.chdir(os.path.join(path, '..'))
import torch
from modelzoo.build_models import BuildModel
from config import get_default_cfg


cfg = get_default_cfg()
cfg.merge_from_files('cfgs/yolox_pd_csp')

builder = BuildModel(cfg)
model = builder.build()

test = torch.ones((2, 3, cfg.data.input_height, cfg.data.input_width))

result = model.core(test)

print(result.shape)

# dataset = CocoDataset('CrowdHuman/annotation_val_coco_style_checked.json', 'CrowdHuman/Images_val', cfg.data, 'val')
# sample = dataset[299]
# show_bbox(sample['img'], sample['anns'])


