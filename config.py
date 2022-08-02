from odcore.config import CN
from odcore.config import get_default_cfg as _get_default_cfg

c = _get_default_cfg()

c.model = CN()
c.model.name = 'yolov3'
c.model.backbone = 'darknet53'
c.model.neck = 'yolov3_neck'
c.model.head = 'yolov3_head'
c.model.use_anchor = True
c.model.fpnlevels = [3, 4, 5]
c.model.anchor_ratios = [2, 4]
c.model.anchor_scales = [0.75, 1]
c.model.assignment_type = 'default'
c.model.assignment_iou_type = 'iou'
c.model.assignment_iou_threshold = 0.6

c.loss = CN()
c.loss.reg_type = ['giou','l1']
c.loss.cls_type = 'bce'
c.loss.use_focal = False
c.loss.focal_alpha = 0.25
c.loss.focal_gamma = 2.0

c.inference = CN()

def get_default_cfg():
    return c.clone()

def get_default_yaml_templete():
    cfg = get_default_cfg()
    cfg.dump_to_file('default_config')

if __name__ == "__main__":
    cfg = get_default_cfg()
    print(type(cfg))
    cfg.merge_from_file('default_config.yaml')
    print(cfg)
    # get_default_yaml_templete()