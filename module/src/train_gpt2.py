import yaml
import tiktoken
import torch
import wandb

from processing_data.dataset import Data
from processing_data.dataloader import get_data_loader
from gpt2 import GPT2Model
from metrics import cross_entropy,accuracy
from trainer import Trainer

with open("config.yaml","r") as f:
    config = yaml.safe_load(f)

with open("generate_text_config.yaml","r") as f:
    generate_text_config = yaml.safe_load(f)


with open("raw_data/the-verdict-train.txt","r") as f: 
    train_text = f.read()
with open("raw_data/the-verdict-val.txt","r") as f: 
    val_text = f.read()





train_dataset = Data(
    raw_text=train_text,
    tokenizer=tiktoken.get_encoding("gpt2"),
    context_length=config["context_window"],
    stride=config["stride"]
)

val_dataset = Data(
    raw_text=val_text,
    tokenizer=tiktoken.get_encoding("gpt2"),
    context_length=config["context_window"],
    stride=config["stride"]
)

train_dl = get_data_loader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=config["shuffle"],
    drop_last=config["drop_last"],
    num_workers=config["num_workers"]
    )

val_dl = get_data_loader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=config["shuffle"],
    drop_last=config["drop_last"],
    num_workers=config["num_workers"]
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
    name="generate text as function",
    config=config
)
    trainer.train()
    wandb.finish()
    torch.save(model.state_dict(), 'model.pth')