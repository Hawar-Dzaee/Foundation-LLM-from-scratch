import os
import yaml
import tiktoken
import wandb
import logging
import torch 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset

torch.set_float32_matmul_precision("high")  # Must come before importing any local modules [says GPT ]


from processing_data.dataset import TinyStoryData
from processing_data.dataloader import get_data_loader,tiny_story_collate
from model_components.gpt2 import GPT2Model
from common.metrics import cross_entropy,accuracy
from common.trainer import Trainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def setup_ddp():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        dist.barrier()
    
    return rank, world_size, local_rank

def cleanup_ddp():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    # Setup distributed training
    rank, world_size, local_rank = setup_ddp()
    
    with open("config.yaml","r") as f:
        config = yaml.safe_load(f)

    with open("generate_text_config.yaml","r") as f:
        generate_text_config = yaml.safe_load(f)

    # Adjust batch size for multi-GPU training
    original_batch_size = config["batch_size"]
    effective_batch_size = original_batch_size 
    config["batch_size"] = original_batch_size  # Per-GPU batch size
    config["effective_batch_size"] = effective_batch_size
    
    if rank == 0:
        logging.info(f"Using {world_size} GPUs")
        logging.info(f"Per-GPU batch size: {config['batch_size']}")
        logging.info(f"Effective batch size: {effective_batch_size}")

    # Create datasets
    train_dataset = TinyStoryData(
        dataset= load_dataset("roneneldan/TinyStories", split="train[:3%]"),
        tokenizer=tiktoken.get_encoding("gpt2"),
        cache_file = "processed_data_train.pt",
        max_length= config["context_window"],
    )

    val_dataset = TinyStoryData(
        dataset= load_dataset("roneneldan/TinyStories", split="train[97%:]"),
        tokenizer=tiktoken.get_encoding("gpt2"),
        cache_file = "processed_data_valid.pt",
        max_length= config["context_window"]
    )

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_dl = get_data_loader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False if world_size > 1 else config["shuffle"],  # Don't shuffle when using DistributedSampler
        drop_last=config["drop_last"],
        num_workers=config["num_workers"],
        collate_fn=tiny_story_collate,
        pin_memory=True,
        sampler=train_sampler
    )

    val_dl = get_data_loader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=config["drop_last"],
        num_workers=config["num_workers"],
        collate_fn=tiny_story_collate,
        pin_memory=True,
        sampler=val_sampler
    )

    # Create model
    model = GPT2Model(config)
    model = model.to(f'cuda:{local_rank}')
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    model = torch.compile(model)

    num_parameters = sum(p.numel() for p in model.parameters())
    if rank == 0:
        logging.info(f"Number of parameters: {num_parameters:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9,0.95), eps=1e-8)

    trainer = Trainer(
        model,
        train_dl,
        val_dl,
        loss_fn=cross_entropy,
        accuracy_fn=accuracy,
        optimizer=optimizer,
        config=config,
        generate_text_config=generate_text_config,
        overfit_single_batch=False
    )

    if rank == 0:
        wandb.init(
            project="Foundation_models",
            name=f"DDP Training - {world_size} GPUs - B:{effective_batch_size}",
            config=config
        )
    
    try:
        trainer.train()
    finally:
        if rank == 0:
            wandb.finish()
        cleanup_ddp()

if __name__ == "__main__":
    main()
