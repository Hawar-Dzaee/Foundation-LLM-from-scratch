import yaml
import tiktoken
import torch
import wandb

from processing_data.dataset import ClassificationDataset
from processing_data.dataloader import get_data_loader
from gpt2 import GPT2Model
from metrics import classification_loss,classification_accuracy
from train import Trainer



with open("config.yaml","r") as f:
    config = yaml.safe_load(f)

with open("generate_text_config.yaml","r") as f:
    generate_text_config = yaml.safe_load(f)

with open("raw_data/the-verdict.txt","r") as f: 
    raw_text = f.read()

config['num_classes'] = 2


train_dateset = ClassificationDataset(
    csv_path='raw_data/sms_spam_collection/train.csv',
    tokenizer=tiktoken.get_encoding("gpt2"),
    max_len=None
)
val_dataset = ClassificationDataset(
    csv_path='raw_data/sms_spam_collection/val.csv',
    tokenizer=tiktoken.get_encoding("gpt2"),
    max_len=train_dateset.max_len
)



train_dl = get_data_loader(
    train_dateset,
    batch_size=config['batch_size'],
    shuffle=config['shuffle'],
    drop_last=config['drop_last'],
    num_workers=config['num_workers']
    )
val_dl = get_data_loader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=config['shuffle'],
    drop_last=config['drop_last'],
    num_workers=config['num_workers']
)


model = GPT2Model(config)
optimizer = torch.optim.AdamW(model.parameters(),lr=0.0004)


wandb.init(
    project="Foundation_models",
    name="classification",
    config=config
)


trainer = Trainer(
    model,
    train_dl,
    val_dl,
    loss_fn=classification_loss,
    accuracy_fn=classification_accuracy,
    optimizer=optimizer,
    config=config,
    device="cpu",
    generate_text_config=generate_text_config
)

if __name__ == "__main__":
    trainer.train(epochs=10,generate_text=False)
    wandb.finish()