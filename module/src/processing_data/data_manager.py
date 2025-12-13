import yaml 
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader,DistributedSampler

from .dataset import TinyStoryData
from .dataloader import tiny_story_collate


with open("config.yaml","r") as f:
    config = yaml.safe_load(f)

def fetch_train_val_dl(): 
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

    print(f'Train Dataset sample size : {len(train_dataset)}')
    print(f'Val Dataset sample size : {len(val_dataset)}')

    train_dl = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        drop_last=config["drop_last"],
        num_workers=config["num_workers"],
        collate_fn=tiny_story_collate,
        sampler = DistributedSampler(train_dataset,shuffle=False)
        )

    val_dl = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        drop_last=config["drop_last"],
        num_workers=config["num_workers"],
        collate_fn=tiny_story_collate,
        sampler = DistributedSampler(val_dataset,shuffle=False)
    )

    return train_dl,val_dl