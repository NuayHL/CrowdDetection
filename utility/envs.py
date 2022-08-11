import os

def get_envs():
    """Get PyTorch needed environments from system envirionments."""
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    return local_rank, rank, world_size