import yaml
import tiktoken
import torch
import wandb
import logging

from datasets import load_dataset

from processing_data.dataset import TinyStoryData
from processing_data.dataloader import get_data_loader,tiny_story_collate
from model_components.gpt2 import GPT2Model
from common.metrics import cross_entropy,accuracy
from common.trainer import Trainer

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
    drop_last=config["drop_last"],
    num_workers=config["num_workers"],
    collate_fn=tiny_story_collate
    )

val_dl = get_data_loader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=config["shuffle"],
    drop_last=config["drop_last"],
    num_workers=config["num_workers"],
    collate_fn=tiny_story_collate
)


model = GPT2Model(config)
import os

# Check if a best model checkpoint exists and load it
best_model_path = "best_model_train_loss.pth"
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path,weights_only=True, map_location=config.get("device", "cpu")))
    logging.info(f"Loaded best model from {best_model_path}")
else:
    logging.info("No best model checkpoint found. Training from scratch.")

num_parameters = sum(p.numel() for p in model.parameters())
logging.info(f"Number of parameters: {num_parameters:,}")

optimizer = torch.optim.AdamW(model.parameters(),lr=config["learning_rate"])





trainer = Trainer(
    model,
    train_dl,
    val_dl,
    loss_fn=cross_entropy,
    accuracy_fn=accuracy,
    optimizer=optimizer,
    config=config,
    generate_text_config=generate_text_config,
    overfit_single_batch= True
)

if __name__ == "__main__":
    wandb.init(
    project="Foundation_models",
    name="alternative batching/Epoch logging",
    config=config
)
    trainer.train()
    wandb.finish()
    torch.save(model.state_dict(), 'final_model.pth')