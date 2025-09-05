import logging
import yaml

import wandb
import torch 
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

from datasets import load_dataset


from processing_data.dataset import TinyStoryData
from processing_data.dataloader import get_data_loader,tiny_story_collate
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



test_dataset = TinyStoryData(
    dataset= load_dataset("roneneldan/TinyStories", split="validation[:1%]"),
    tokenizer=tiktoken.get_encoding("gpt2"),
    cache_file = "processed_data_test.pt",
    max_length= config["context_window"],
)

test_loader = get_data_loader(
    test_dataset,
    batch_size=config["batch_size"],
    shuffle=config["shuffle"],
    drop_last=config["drop_last"],
    num_workers=config["num_workers"],
    collate_fn=tiny_story_collate
)


model = GPT2Model(config)
loaded_weights = torch.load('best_model_val_loss.pth',weights_only=True)
model.load_state_dict(loaded_weights)


evaluator = Evaluator(
    model,
    test_dl = test_loader,
    loss_fn = cross_entropy,
    accuracy_fn = accuracy,
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    )




if __name__ == "__main__":
    wandb.init(
        project="Foundation_models",
        name="evaluating gpt2 stories on GPU",
        config=config
    )
    evaluator.evaluate()
    wandb.finish()