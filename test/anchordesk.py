import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path, 'odcore'))
import torch
import cv2
import numpy as np
from odcore.utils.visualization import printImg, show_bbox, _add_bbox_img, _add_point_img
from utility.anchors import generateAnchors, anchors_parse, Anchor
from config import get_default_cfg

cfg = get_default_cfg()
cfg.merge_from_file('cfgs/anchortest.yaml')

width = cfg.data.input_width
height = cfg.data.input_height

gray_img = np.ones((height*2,width*2,3),dtype=np.int32)*191
gray_img[int(height/2):int(height*3/2), int(width/2):int(width*3/2), :] -= 64

anchors = Anchor(cfg)
points = anchors.gen_points(singleBatch=True)
print(points)
points[:,0] += width/2
points[:,1] += height/2
gray_img = _add_point_img(gray_img, points)
printImg(gray_img)

# bbox = anchors.gen_Bbox(singleBatch=True)
# parsed_anchors = anchors_parse(config=cfg, anchors=bbox)
#
# # gray_img = _add_bbox_img(gray_img, parsed_anchors[0][1], type='xywh')
# # gray_img = _add_bbox_img(gray_img, parsed_anchors[0][3], color=[255,0,0], type='xywh')
# # printImg(gray_img)
# show_bbox(gray_img, parsed_anchors[0][0],type='xywh')

