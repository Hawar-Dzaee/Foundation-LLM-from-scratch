import os

import yaml
import tiktoken
import torch
import wandb
import logging


# torch.set_float32_matmul_precision("high")  # Must come before importing any local modules [says GPT ]


import platform 
from torch.utils.data.distributed import DistributedSampler 
from torch.distributed import init_process_group, destroy_process_group 
import torch.distributed as dist




from datasets import load_dataset


from processing_data.dataset import TinyStoryData
from processing_data.dataloader import get_data_loader,tiny_story_collate
from model_components.gpt2 import GPT2Model
from common.metrics import cross_entropy,accuracy
from common.trainer import Trainer


# torch.set_float32_matmul_precision("high")  # position 2 : No difference with Postion 1 (P2 was 2 seconds faster than P1 :negligble)



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

with open("config.yaml","r") as f:
    config = yaml.safe_load(f)

with open("generate_text_config.yaml","r") as f:
    generate_text_config = yaml.safe_load(f)


def setup_distributed():
    """Initialize distributed training if needed"""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        # Initialize process group
        init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return True
    return False


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        destroy_process_group()


#========================================
def main():
    # Get DDP parameters from environment variables set by torchrun
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"DEBUG: Environment variables:")
    print(f"  RANK={os.environ.get('RANK', 'Not set')}")
    print(f"  LOCAL_RANK={os.environ.get('LOCAL_RANK', 'Not set')}")
    print(f"  WORLD_SIZE={os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"  MASTER_ADDR={os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"  MASTER_PORT={os.environ.get('MASTER_PORT', 'Not set')}")
    print(f"Running on rank {rank}/{world_size}, local_rank {local_rank}")
    print(f"Available GPUs: {torch.cuda.device_count()}")


     # Initialize distributed training if needed
    is_distributed = setup_distributed()

    # Set device based on local_rank
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        config['device'] = device  # Update config with correct device
        print(f"Using device: {device}")
    else:
        device = "cpu"
        config['device'] = device
        print(f"Using device: {device}")


    if rank == 0:
        wandb.init(
            project="Foundation_models",
            name="Single GPU (10 percent data)",
            config=config
        )


    model = GPT2Model(config)
    # model = torch.compile(model)

    num_parameters = sum(p.numel() for p in model.parameters())
    if rank == 0 : 
        logging.info(f"Number of parameters: {num_parameters:,}")


    optimizer = torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],betas = (0.9,0.95),eps=1e-8)


    train_dataset = TinyStoryData(
        dataset= load_dataset("roneneldan/TinyStories", split="train[:10%]"),
        tokenizer=tiktoken.get_encoding("gpt2"),
        cache_file = "processed_data_train.pt",
        max_length= config["context_window"],

    )

    val_dataset = TinyStoryData(
        dataset= load_dataset("roneneldan/TinyStories", split="train[90%:]"),
        tokenizer=tiktoken.get_encoding("gpt2"),
        cache_file = "processed_data_valid.pt",
        max_length= config["context_window"]
    )


    # is_distributed = world_size > 1 
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    train_dl = get_data_loader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        pin_memory = True ,
        drop_last=config["drop_last"],
        # num_workers=config["num_workers"],
        collate_fn=tiny_story_collate,
        sampler = train_sampler
        )

    val_dl = get_data_loader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        pin_memory = True,
        drop_last=config["drop_last"],
        # num_workers=config["num_workers"],
        collate_fn=tiny_story_collate,
        sampler = val_sampler
    )



    trainer = Trainer(
        model,
        train_dl,
        val_dl,
        loss_fn=cross_entropy,
        accuracy_fn=accuracy,
        optimizer=optimizer,
        config=config,
        generate_text_config=generate_text_config,
        overfit_single_batch= False,
        rank = None, 
        world_size = None 
    )

    history = trainer.train()

    if rank == 0 : 
        wandb.finish()
        model_to_save = trainer.model.module if hasattr(trainer.model,"module") else trainer.model
        torch.save(model_to_save.state_dict(), 'final_model.pth')

    cleanup_distributed()
    
    return history

if __name__ == "__main__":
    main()

