import math
from collections import defaultdict
import torch.nn as nn
import modelzoo.head as head
import modelzoo.neck as neck
import modelzoo.backbone as backbone

from modelzoo.yolov3 import Yolov3
from modelzoo.yolo import YoloX
from modelzoo.retinanet import RetinaNet
from modelzoo.pdyolo import PDYOLO

class BuildModel():
    def __init__(self, config):
        self.config = config
        self.model_name = config.model.name.lower()
    def build(self):
        print("Building Models: %s"%self.model_name)
        main_model = self.get_model_structure()
        backbone_name = self.config.model.backbone.lower()
        neck_name = self.config.model.neck.lower()
        head_name = self.config.model.head.lower()
        model_backbone, p3c = backbone.build_backbone(backbone_name)
        model_neck, p3c_r = neck.build_neck(neck_name)
        model_head = head.build_head(head_name)

        main_model_dict = {}
        backbone_dict = {}
        neck_dict = {}
        head_dict = {}
        if self.config.model.structure_extra != None:
            sdict = defaultdict(dict,self.config.model.structure_extra[0])
            main_model_dict = sdict['model']
            backbone_dict = sdict['backbone']
            neck_dict = sdict['neck']
            head_dict = sdict['head']

        if 'cspdarknet' in backbone_name:
            p3c = int(64 * backbone_dict['width']) * 4

        classes = self.config.data.numofclasses
        anchors = 1
        if self.config.model.use_anchor:
            anchors = len(self.config.model.anchor_ratios[0])

        model = main_model(self.config,
                           backbone=model_backbone(**backbone_dict),
                           neck=model_neck(p3c, **neck_dict),
                           head=model_head(classes, anchors, int(p3c * p3c_r), **head_dict),
                           **main_model_dict)

        # weight init
        model.apply(weight_init(self.config))
        model.head.apply(head_bias_init(0.01))

        print("Num of Parameters: %.2fM"%(numofParameters(model)/1e6))
        return model

    def get_model_structure(self):
        if self.model_name == 'yolov3':
            return Yolov3
        elif self.model_name == 'retinanet':
            return RetinaNet
        elif self.model_name in ['yolox', 'yolo_like']:
            return YoloX
        elif self.model_name == 'pdyolo':
            return PDYOLO
        else:
            raise NotImplementedError('No model named %s' % (self.model_name))

def weight_init(config):
    def init_func(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(config.model.init.bn_weight)
            m.bias.data.fill_(config.model.init.bn_bias)
            m.eps = config.model.init.bn_eps
            m.momentum = config.model.init.bn_momentum
    return init_func

def head_bias_init(prior_prob):
    def init_func(m):
        fill_in_ = -math.log((1 - prior_prob) / prior_prob)
        if isinstance(m, nn.Conv2d):
            if m.bias != None:
                nn.init.constant_(m.bias, fill_in_)
    return init_func

def numofParameters(model: nn.Module ):
    nump = 0
    for par in model.parameters():
        nump += par.numel()
    return nump

if __name__ == '__main__':
    import sys
    import os
    path = os.getcwd()
    sys.path.append(os.path.join(path, 'odcore'))

    import torch
    from config import get_default_cfg
    from odcore.config import merge_from_files
    config = get_default_cfg()
    merge_from_files(config, 'cfgs/yolox')
    builder = BuildModel(config)
    model = builder.build().cuda()
    input = torch.rand((1,3,640,640)).cuda()
    output = model.core(input)
    print(output.shape)