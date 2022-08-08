import sys
import os
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
import torch
from odcore.engine.train import Train
from odcore.args import get_train_args_parser
from config import get_default_cfg
from modelzoo.build_models import BuildModel

def main(args):
    config = get_default_cfg()
    if args.resume_exp != '':
        print('Resume Training')
        filelist = os.listdir(args.resume_exp)
        cfg_flag = False
        for files in filelist:
            if 'cfg.yaml' in files:
                cfg_file_name = files
                print('Find %s'%cfg_file_name)
                config.merge_from_file(os.path.join(args.resume_exp, cfg_file_name))
                cfg_flag = True
                break
        if not cfg_flag:
            raise FileNotFoundError('Cannot find cfg file')
    else:
        config.merge_from_file(args.conf_file)
        config.training.batch_size = args.batch_size
        config.training.workers = args.workers
        config.training.eval_interval = args.eval_interval
        print('Normal Training')
    builder = BuildModel(config)
    model = builder.build()
    if args.device == '0':
        rank = -1
    else:
        raise NotImplementedError
    train = Train(config,args,model,rank)
    train.go()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = get_train_args_parser().parse_args()
    main(args)