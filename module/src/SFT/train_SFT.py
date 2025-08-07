import json

import yaml
import tiktoken
import torch
import wandb
import logging

from processing_data.dataset import InstructionDataset
from processing_data.dataloader import get_data_loader,instruction_collate_fn

from model_components.gpt2 import GPT2Model
from common.metrics import cross_entropy,accuracy
from common.trainer import Trainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

tokenizer = tiktoken.get_encoding("gpt2")

with open("raw_data/instruction-examples.json","r") as f:
    data = json.load(f)

train_index = int(len(data) * 0.8)
val_index = int(len(data) * 0.1)

train_data = data[:train_index]
val_data = data[train_index: train_index + val_index]
test_data = data[train_index + val_index:]



with open("config.yaml","r") as f:
    config = yaml.safe_load(f)

with open("chat_completion_config.yaml","r") as f:
    chat_config = yaml.safe_load(f)







train_ds = InstructionDataset(train_data,tokenizer)
val_ds = InstructionDataset(val_data,tokenizer)


train_dl = get_data_loader(
    train_ds,
    batch_size=2,
    shuffle=False,
    drop_last=True,
    num_workers=0,
    collate_fn=instruction_collate_fn
    )

val_dl = get_data_loader(
    val_ds,
    batch_size=2,
    shuffle=False,
    drop_last=True,
    num_workers=0,
    collate_fn=instruction_collate_fn
    )


model = GPT2Model(config)
loaded_weights = torch.load('model.pth')
model.load_state_dict(loaded_weights)
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
    generate_text_config=chat_config
)

if __name__ == "__main__":
    wandb.init(
    project="Foundation_models",
    name="SFT adding inference file",
    config=config
)
    trainer.train()
    wandb.finish()
    torch.save(model.state_dict(), 'sft_model.pth')