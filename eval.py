import os
import sys
from odcore.engine.eval import Eval
from config import get_default_cfg
from odcore.args import get_eval_args_parser
from modelzoo.build_models import BuildModel
from tools.coco2odgt import coco2odgt_det

def main(args):
    cfg = get_default_cfg()
    cfg.merge_from_files(args.conf_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    builder = BuildModel(cfg)
    model = builder.build()
    model.set(args, 0)
    flag = 0
    if args.type == 'mip':
        print('using mip for eval, firstly eval mr')
        args.type = 'mr'
        flag = 1
    evaluate = Eval(cfg, args, model, 0)
    evaluate.eval()
    if flag == 1:
        json_pre_file = os.path.splitext(args.ckpt_file)[0] + '_evalresult.json'
        odgt_file = coco2odgt_det(json_pre_file)
        os.system('python evaluate/compute_APMR.py --detfile %s --target_key box' % odgt_file)
        os.system('python evaluate/compute_JI.py --detfile %s --target_key box' % odgt_file)

if __name__ == "__main__":
    print(os.getcwd())
    args = get_eval_args_parser().parse_args()
    main(args)