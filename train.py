import sys
import os
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from odcore.engine.train import Train
from odcore.args import get_train_args_parser
from config import get_default_cfg
from utility.envs import get_envs
from modelzoo.build_models import BuildModel

def main(args):
    args.rank, args.local_rank, args.world_size = get_envs()
    is_mainprocess = True if args.local_rank in [-1,0] else False
    config = get_default_cfg()
    if args.resume_exp != '':
        if is_mainprocess: print('Resume Training')
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
        config.training.batch_size = int(args.batch_size/args.world_size)
        config.training.workers = int(args.workers/args.world_size)
        config.training.eval_interval = args.eval_interval
        if is_mainprocess: print('Normal Training')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    available_device = [int(device) for device in args.device.strip().split(',')]
    if args.local_rank != -1: # if DDP mode
        if is_mainprocess: print('Initializing process group... ')
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                                rank=args.local_rank, world_size=args.world_size)
    builder = BuildModel(config)
    model = builder.build()
    train = Train(config,args,model,args.local_rank)
    train.go()

if __name__ == '__main__':
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = get_train_args_parser().parse_args()
    main(args)