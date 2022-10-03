import sys
import os
path = os.getcwd()
sys.path.append(os.path.join(path,'odcore'))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from odcore.engine.train import Train as _Train
from odcore.args import get_train_args_parser
from config import get_default_cfg
from utility.envs import get_envs, seed_init
from modelzoo.build_models import BuildModel


class Train():
    def __init__(self, args, rank, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.is_mainprocess = True if rank in [-1,0] else False
        self.init_process_group()
        self.args = args
        self.config = get_default_cfg()

        self.check_resume()
        self.builder = BuildModel(self.config)

    def go(self):
        seed_init(self.config.seed)
        model = self.builder.build()
        train = _Train(self.config, self.args, model, self.rank)
        train.go()

    def check_resume(self):
        if self.args.resume_exp != '':
            if self.is_mainprocess: print('Resume Training')
            self.load_pre_exp_cfgs()
        else:
            if self.is_mainprocess: print('Normal Training')
            self.config.merge_from_files(self.args.conf_file)
            self.syn_config_with_args()

    def load_pre_exp_cfgs(self):
        filelist = os.listdir(self.args.resume_exp)
        cfg_flag = False
        for files in filelist:
            if 'cfg.yaml' in files:
                cfg_file_name = files
                print('Find %s' % cfg_file_name)
                self.config.merge_from_file(os.path.join(self.args.resume_exp, cfg_file_name))
                cfg_flag = True
                break
        if not cfg_flag:
            raise FileNotFoundError('Cannot find cfg file')

    def syn_config_with_args(self):
        self.config.training.batch_size = int(self.args.batch_size / self.world_size)
        self.config.training.workers = int(self.args.workers / self.world_size)
        self.config.training.eval_interval = self.args.eval_interval
        self.config.training.accumulate = self.args.accumu

    def init_process_group(self):
        if self.rank != -1:  # if DDP mode
            if self.is_mainprocess: print('Initializing process group... ')
            dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                                    rank=self.rank, world_size=self.world_size)


def init_training(rank, args, world_size):
    train = Train(args, rank, world_size)
    train.go()


def main(args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    single_training = True
    if args.device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        available_device = [int(device) for device in args.device.strip().split(',')]
        world_size = len(available_device)
        if world_size != 1:
            single_training = False
    if single_training:
        init_training(-1, args, 1)
    else:
        mp.spawn(init_training, (args, world_size), nprocs=world_size)

if __name__ == '__main__':
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = get_train_args_parser().parse_args()
    main(args)