import sys
import os
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
from odcore.engine.train import Train
from odcore.args import get_args_parser
from config import get_default_cfg
from modelzoo.build_models import BuildModel

def main(args):
    config = get_default_cfg()
    config.merge_from_file(args.conf_file)
    config.training.batch_size = args.batch_size
    builder = BuildModel(config)
    model = builder.build()
    if args.device == '0':
        rank = -1
    else:
        raise NotImplementedError
    train = Train(config,args,model,rank)
    train.go()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = get_args_parser().parse_args()
    main(args)