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
c.loss.weight = [3.0, 10.0, 5.0] #reg, cls, obj

c.inference = CN()
c.inference.nms_type = 'nms'
c.inference.obj_thres = 0.7
c.inference.iou_thres = 0.7

def get_default_cfg():
    return c.clone()

def get_default_yaml_templete():
    cfg = get_default_cfg()
    cfg.dump_to_file('default_config')

def updata_config_file(filepath):
    cfg = get_default_cfg()
    cfg.merge_from_file(filepath)
    cfg.dump_to_file(filepath[:-5])

if __name__ == "__main__":
    updata_config_file('yolo_ch.yaml')