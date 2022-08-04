import sys
import os
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
from odcore.data.dataloader import build_dataloader
from odcore.data.dataset import CocoDataset
from config import get_default_cfg
from odcore.utils.visualization import show_bbox, dataset_inspection
import numpy as np
import torch


cfg = get_default_cfg()
cfg.merge_from_file('test_config.yaml')
dataset = CocoDataset('CrowdHuman/annotation_train_coco_style_100.json','CrowdHuman/Images_train',config_data=cfg.data, task='train')
loader = build_dataloader('CrowdHuman/annotation_train_coco_style_100.json','CrowdHuman/Images_train',config_data=cfg.data,
                          batch_size=1, rank=-1,workers=0, task='train')
dataset_inspection(dataset, 10)
dataset_inspection(dataset, 10)
flag = 1
while (flag):
    for samples in loader:
        if flag == 0: break
        for idx, anns in enumerate(samples['annss']):
            print('fin:',np.greater(anns[:,4], -0.5).sum())
            if np.greater(anns[:,4], -0.5).sum() > 0:
                img = samples['imgs'][idx].numpy()
                img = np.transpose(img,(1,2,0))
                show_bbox(img, anns)
                flag = 0
                break
