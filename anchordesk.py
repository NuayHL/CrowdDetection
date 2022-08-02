import torch
import cv2
import numpy as np
from odcore.utils.visualization import printImg, show_bbox, _add_bbox_img
from utility.anchors import generateAnchors, anchors_parse
from config import get_default_cfg

cfg = get_default_cfg()

width = cfg.data.input_width
height = cfg.data.input_height

gray_img = np.ones((height*2,width*2,3),dtype=np.int32)*191
gray_img[int(height/2):int(height*3/2), int(width/2):int(width*3/2), :] -= 64

anchors = generateAnchors(config=cfg, singleBatch=True)
parsed_anchors = anchors_parse(config=cfg, anchors=anchors)

gray_img = _add_bbox_img(gray_img, parsed_anchors[2][1], type='xywh')
gray_img = _add_bbox_img(gray_img, parsed_anchors[2][3], color=[255,0,0], type='xywh')
printImg(gray_img)
#show_bbox(gray_img, parsed_anchors[2][0],type='x1y1x2y2')

