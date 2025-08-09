import yaml
import tiktoken
import torch
import wandb

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





dataset = load_dataset("roneneldan/TinyStories")


train_dataset = TinyStoryData(
    dataset= dataset,
    split ="train",
    tokenizer=tiktoken.get_encoding("gpt2"),
    max_length= config["context_window"]
)

val_dataset = TinyStoryData(
    dataset= dataset,
    split ="validation",
    tokenizer=tiktoken.get_encoding("gpt2"),
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
optimizer = torch.optim.AdamW(model.parameters(),lr=0.0004)





trainer = Trainer(
    model,
    train_dl,
    val_dl,
    loss_fn=cross_entropy,
    accuracy_fn=accuracy,
    optimizer=optimizer,
    config=config,
    device="cpu",
    generate_text_config=generate_text_config
)

if __name__ == "__main__":
    wandb.init(
    project="Foundation_models",
    name="changing folder diretory",
    config=config
)
    trainer.train()
    wandb.finish()
    torch.save(model.state_dict(), 'model.pth')