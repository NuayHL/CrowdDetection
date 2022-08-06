import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
from odcore.engine.infer import Infer
from config import get_default_cfg
from odcore.args import get_infer_args_parser
from odcore.utils.visualization import show_bbox
from modelzoo.build_models import BuildModel


def main(args):
    cfg = get_default_cfg()
    cfg.merge_from_file(args.conf_file)
    builder = BuildModel(cfg)
    model = builder.build()
    model.set(args, 0)
    infer = Infer(cfg, args, model, 0)
    result, img = infer(args.img)
    result = result[0].to_ori_label(experiment_input_shape = (cfg.data.input_width, cfg.data.input_height))
    for i in result[:,4]:
        print('score:', i)
    show_bbox(img, result, type='x1y1x2y2')


if __name__ == "__main__":
    args = get_infer_args_parser().parse_args()
    main(args)
    a = 'IMG/IMG_20220609_141503.jpg'