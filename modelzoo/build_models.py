import torch.nn as nn
import modelzoo.head as head
import modelzoo.neck as neck
import modelzoo.backbone as backbone
from modelzoo.yolov3 import Yolov3

class BuildModel():
    def __init__(self, config):
        self.config = config
        self.model_name = config.model.name.lower()
    def build(self):
        print("Building Models: %s"%self.model_name)
        main_model = self.get_model_structure()
        model_backbone, p3c = backbone.build_backbone(self.config.model.backbone.lower())
        model_neck, p3c_r = neck.build_neck(self.config.model.neck.lower())
        model_head = head.build_head(self.config.model.head.lower())

        classes = self.config.data.numofclasses
        if self.config.model.use_anchor:
            anchors_per_grid = len(self.config.model.anchor_ratios) * len(self.config.model.anchor_scales)
            model_head = model_head(classes, anchors_per_grid, int(p3c * p3c_r))
        else:
            model_head = model_head(classes, int(p3c * p3c_r))

        model = main_model(self.config, backbone=model_backbone(), neck=model_neck(p3c), head=model_head)
        # weight init
        model.apply(weight_init)
        return model

    def get_model_structure(self):
        if self.model_name == 'yolov3':
            return Yolov3
        else:
            raise NotImplementedError('No model named %s' % (self.model_name))

def weight_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if 'BatchNorm' in classname:
        m.weight.data.fill_(1)
        m.bias.data.zero_()

if __name__ == '__main__':
    import torch
    from config import get_default_cfg
    config = get_default_cfg()
    builder = BuildModel(config)
    model = builder.build().cuda()
    input = torch.rand((1,3,640,640)).cuda()
    input = {'imgs':input}
    output = model.core(input)
    print(output[0].shape,output[1].shape,output[2].shape)