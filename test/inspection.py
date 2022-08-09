import sys
import os
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
from odcore.data.dataloader import build_dataloader
from odcore.data.dataset import CocoDataset
from odcore.data.data_augment import *
from config import get_default_cfg
from odcore.utils.visualization import show_bbox, dataset_inspection
import numpy as np
import torch


cfg = get_default_cfg()
cfg.merge_from_file('coco_config.yaml')
dataset = CocoDataset('COCO/instances_train2017.json','COCO/train2017',config_data=cfg.data, task='train')
# loader = build_dataloader('CrowdHuman/annotation_train_coco_style_100.json','CrowdHuman/Images_train',config_data=cfg.data,
#                           batch_size=1, rank=-1,workers=0, task='train')
# samples = dataset.load_sample(10)
# show_bbox(samples['img'],samples['anns'])
# letterbox = LetterBox(cfg.data, auto=False, scaleup=True)
# letterbox(samples)
# show_bbox(samples['img'],samples['anns'])
# ra = RandomAffine(cfg.data)
# ra.img_size = (320,320)
# ra(samples)
# show_bbox(samples['img'],samples['anns'])

for anns_id in dataset.annotations.anns:
    bbox = np.array(dataset.annotations.anns[anns_id]['bbox'])
    print(bbox)
    if bbox.shape[0] == 0:
        print('TMD')

# dataset_inspection(dataset, 15, anntype='xywh')
# dataset_inspection(dataset, 15, anntype='xywh')
# dataset_inspection(dataset, 15, anntype='xywh')
# dataset_inspection(dataset, 15, anntype='xywh')
# flag = 1
# while (flag):
#     for samples in loader:
#         if flag == 0: break
#         for idx, anns in enumerate(samples['annss']):
#             print('fin:',np.greater(anns[:,4], -0.5).sum())
#             if np.greater(anns[:,4], -0.5).sum() == 0:
#                 img = samples['imgs'][idx].numpy()
#                 img = np.transpose(img,(1,2,0))
#                 show_bbox(img, anns)
#                 flag = 0
#                 break
