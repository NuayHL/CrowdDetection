import os
import sys

path = os.getcwd()
sys.path.append(os.path.join(path, 'odcore'))
from infer import VideoInfer
from config import get_default_cfg
from odcore.args import get_infer_args_parser


def main(args):
    assert os.path.exists(args.source), 'Invalid source path'
    cfg = get_default_cfg()
    cfg.merge_from_files(args.conf_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    infer = VideoInfer(cfg, args, 0)
    infer.infer()


if __name__ == "__main__":
    args = get_infer_args_parser().parse_args()
    main(args)
