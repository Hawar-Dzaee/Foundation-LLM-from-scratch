import os 
import torch 
from torch.distributed import init_process_group

def ddp_setup(rank,world_size):
    if 'MASTER_ADDR' not in os.environ : 
        os.environ['MASTER_ADDR'] = "localhost"
    
    if "MASTER_PORT" not in os.environ : 
        os.environ['MASTER_PORT'] = "12345"


    init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
        )

    torch.cuda.set_device(rank)