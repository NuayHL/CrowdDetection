import os
path = os.getcwd()
os.chdir(os.path.join(path, '..'))
from odcore.data.dataset import CocoDataset
from config import get_default_cfg

cfg = get_default_cfg()
cfg.merge_from_files('cfgs/yolox_ori')

dataset = CocoDataset(cfg.training.val_img_anns_path, cfg.training.val_img_path, cfg.data, 'val')

print(len(dataset.annotations.anns))