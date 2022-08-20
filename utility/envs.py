import os
import numpy as np
import torch

def get_envs():
    """Get PyTorch needed environments from system envirionments."""
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    return local_rank, rank, world_size

def seed_init(num=None):
    if num == None: return 0
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)