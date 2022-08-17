import sys
import os
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
import torch
import torch.distributed as dist
from odcore.engine.train import Train as _Train
from odcore.args import get_train_args_parser
from config import get_default_cfg
from utility.envs import get_envs
from modelzoo.build_models import BuildModel

class Train():
    def __init__(self, args):
        self.args = args
        args.rank, args.local_rank, args.world_size = get_envs()
        self.is_mainprocess = True if args.local_rank in [-1,0] else False
        self.config = get_default_cfg()
        self.set_available_device()
        self.init_process_group()
        self.check_resume()
        self.builder = BuildModel(self.config)

    def go(self):
        model = self.builder.build()
        train = _Train(self.config, self.args, model, self.args.local_rank)
        train.go()

    def check_resume(self):
        if args.resume_exp != '':
            if self.is_mainprocess: print('Resume Training')
            filelist = os.listdir(args.resume_exp)
            cfg_flag = False
            for files in filelist:
                if 'cfg.yaml' in files:
                    cfg_file_name = files
                    print('Find %s' % cfg_file_name)
                    self.config.merge_from_file(os.path.join(args.resume_exp, cfg_file_name))
                    cfg_flag = True
                    break
            if not cfg_flag:
                raise FileNotFoundError('Cannot find cfg file')
        else:
            if self.is_mainprocess: print('Normal Training')
            self.config.merge_from_file(args.conf_file)
            self.config.training.batch_size = int(args.batch_size / args.world_size)
            self.config.training.workers = int(args.workers / args.world_size)
            self.config.training.eval_interval = args.eval_interval

    def set_available_device(self):
        if not self.is_mainprocess: return
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.device
        self.available_device = [int(device) for device in args.device.strip().split(',')]

    def init_process_group(self):
        if self.args.local_rank != -1:  # if DDP mode
            if self.is_mainprocess: print('Initializing process group... ')
            dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                                    rank=args.local_rank, world_size=args.world_size)

def main(args):
    train = Train(args)
    train.go()

if __name__ == '__main__':
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = get_train_args_parser().parse_args()
    main(args)