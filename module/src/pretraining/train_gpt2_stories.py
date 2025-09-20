import os

import yaml
import tiktoken
import torch
import wandb
import logging


torch.set_float32_matmul_precision("high")  # Must come before importing any local modules [says GPT ]


import platform 
from torch.utils.data.distributed import DistributedSampler 
from torch.distributed import init_process_group, destroy_process_group 




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




train_dataset = TinyStoryData(
    dataset= load_dataset("roneneldan/TinyStories", split="train[:1%]"),
    tokenizer=tiktoken.get_encoding("gpt2"),
    cache_file = "processed_data_train.pt",
    max_length= config["context_window"],

)

val_dataset = TinyStoryData(
    dataset= load_dataset("roneneldan/TinyStories", split="train[99%:]"),
    tokenizer=tiktoken.get_encoding("gpt2"),
    cache_file = "processed_data_valid.pt",
    max_length= config["context_window"]
)

train_dl = get_data_loader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=config["shuffle"],
    pin_memory = True ,
    drop_last=config["drop_last"],
    # num_workers=config["num_workers"],
    collate_fn=tiny_story_collate,
    sampler = DistributedSampler(train_dataset)
    )

val_dl = get_data_loader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=config["shuffle"],
    drop_last=config["drop_last"],
    # num_workers=config["num_workers"],
    collate_fn=tiny_story_collate
)


model = GPT2Model(config)


def main_ddp(
    rank,
    world_size,
    config,
    model,
    train_dataset,
    val_dataset,
    loss_fn,
    accuracy_fn,
    optimizer,
    generate_text_config
):
pass 


# model = torch.compile(model)

num_parameters = sum(p.numel() for p in model.parameters())
logging.info(f"Number of parameters: {num_parameters:,}")

optimizer = torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],betas = (0.9,0.95),eps=1e-8)





trainer = Trainer(
    model,
    train_dl,
    val_dl,
    loss_fn=cross_entropy,
    accuracy_fn=accuracy,
    optimizer=optimizer,
    config=config,
    generate_text_config=generate_text_config,
    overfit_single_batch= False
)

if __name__ == "__main__":


    if "WORLD_SIZE" in os.environ : 
        world_size = int(os.environ["WORLD_SIZE"])\
    else : 
        world_size = 1 

    if "LOCAL_RANK" in os.environ: 
        rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ: 
        rank = int(os.environ["RANK"])
    else : 
        rank = 0 


    if rank == 0: 
        print("Pytorch version:",torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("Number of GPUs available:",torch.cuda.device_count())

    torch.manual_seed(123)

    wandb.init(
        project="Foundation_models",
        name="Global Gradient Clipping",
        config=config
    )
    trainer.train()
    wandb.finish()
    torch.save(model.state_dict(), 'final_model.pth')