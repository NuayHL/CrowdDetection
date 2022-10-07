import numpy as np
from odcore.data.dataset import CocoDataset

def cal_anchors(config):
    fpnlevels = config.model.fpnlevels
    dataset = CocoDataset(config.training.train_img_anns_path,
                          config.trainimg.train_img_path,
                          config.data, 'val')
