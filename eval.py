import os
import sys
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
from odcore.engine.eval import Eval
from config import get_default_cfg
from odcore.args import get_eval_args_parser
from utility.result import result_parse
from modelzoo.build_models import BuildModel

def main(args):
    cfg = get_default_cfg()
    cfg.merge_from_file(args.conf_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    builder = BuildModel(cfg)
    model = builder.build()
    model.set(args, 0)
    eval = Eval(cfg, args, model, 0)
    eval.eval(result_parse)


if __name__ == "__main__":
    args = get_eval_args_parser().parse_args()
    main(args)