import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
import cv2
import matplotlib.pyplot as plt
from odcore.engine.infer import Infer
from config import get_default_cfg
from odcore.args import get_infer_args_parser
from modelzoo.build_models import BuildModel


def main(args):
    assert os.path.exists(args.img),'Invalid image path'
    cfg = get_default_cfg()
    cfg.merge_from_files(args.conf_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    builder = BuildModel(cfg)
    model = builder.build()
    model.set(args, 0)
    infer = Infer(cfg, args, model, 0)
    result, img = infer(args.img, "_test_hot_map")
    upsampled = []
    for id, level in enumerate(result):
        for fm in level:
            fm = fm.numpy()
            upsampled.append(cv2.in)
            cv2.imshow('test',fm)
            cv2.waitKey()


if __name__ == "__main__":
    args = get_infer_args_parser().parse_args()
    main(args)