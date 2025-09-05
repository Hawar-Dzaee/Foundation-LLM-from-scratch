import logging
import yaml

import wandb
import torch 
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from processing_data.dataloader import get_data_loader
from processing_data.dataset import Data
from model_components.gpt2 import GPT2Model
from common.evaluator import Evaluator
from common.metrics import accuracy, cross_entropy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


with open("raw_data/the-verdict-test.txt","r") as f: 
    test_text = f.read()

test_dataset = Data(
    raw_text=test_text,
    tokenizer=tiktoken.get_encoding("gpt2"),
    context_length=config["context_window"],
    stride=config["stride"],
)

test_loader = get_data_loader(
    test_dataset,
    batch_size=config["batch_size"],
    shuffle=config["shuffle"],
    drop_last=config["drop_last"],
    num_workers=config["num_workers"],
)


model = GPT2Model(config)
loaded_weights = torch.load('model.pth')
model.load_state_dict(loaded_weights)


evaluator = Evaluator(
    model,
    test_dl = test_loader,
    loss_fn = cross_entropy,
    accuracy_fn = accuracy,
    device = 'cpu'
    )




if __name__ == "__main__":
    wandb.init(
        project="Foundation_models",
        name="evaluating gpt2 adding inference file",
        config=config
    )
    evaluator.evaluate()
    wandb.finish()