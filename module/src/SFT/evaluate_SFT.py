import json

import yaml
import tiktoken
import torch
import wandb
import logging
from pathlib import Path


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


script_dir = Path(__file__).parent
json_file_path = script_dir / "instruction-examples.json"

with open(json_file_path, "r") as f:
    data = json.load(f)

# with open("instruction-examples.json","r") as f:
#     data = json.load(f)


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
loaded_weights = torch.load('best_model_val_loss.pth')
model.load_state_dict(loaded_weights)

evaluator = Evaluator(
    model,
    test_dl = test_dl,
    loss_fn = cross_entropy,
    accuracy_fn = accuracy,
    device = 'cuda'
    )




if __name__ == "__main__":
    wandb.init(
        project="Foundation_models",
        name="evaluating SFT summary",
        config=config
    )
    evaluator.evaluate()
    wandb.finish()