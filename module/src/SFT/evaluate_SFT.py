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
from common.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open("raw_data/instruction-examples.json","r") as f:
    data = json.load(f)


tokenizer = tiktoken.get_encoding("gpt2")



train_index = int(len(data) * 0.8)
val_index = int(len(data) * 0.1)
test_data = data[train_index + val_index:]



test_ds = InstructionDataset(test_data,tokenizer)


test_dl = get_data_loader(
    test_ds,
    batch_size=2,
    shuffle=False,
    drop_last=True,
    num_workers=0,
    collate_fn=instruction_collate_fn
    )



model = GPT2Model(config)
loaded_weights = torch.load('sft_model.pth')
model.load_state_dict(loaded_weights)

evaluator = Evaluator(
    model,
    test_dl = test_dl,
    loss_fn = cross_entropy,
    accuracy_fn = accuracy,
    device = 'cpu'
    )




if __name__ == "__main__":
    wandb.init(
        project="Foundation_models",
        name="evaluating SFT summary",
        config=config
    )
    evaluator.evaluate()
    wandb.finish()