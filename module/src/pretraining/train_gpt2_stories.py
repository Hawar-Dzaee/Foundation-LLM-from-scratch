import os 
import yaml
import torch
import wandb
import logging
import argparse 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import destroy_process_group 

torch.set_float32_matmul_precision("high")  # Must come before importing any local modules [says GPT ]


from processing_data.data_manager import fetch_train_val_dl
from model_components.gpt2 import GPT2Model
from common.metrics import cross_entropy,accuracy
from common.trainer import Trainer
from distributed import ddp_setup


# torch.set_float32_matmul_precision("high")  # position 2 : No difference with Postion 1 (P2 was 2 seconds faster than P1 :negligble)



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",type=str,required=True,help="Name of Wandb run")
    return parser.parse_args()


def main(rank,world_size) : 
    args = parse_args()
    ddp_setup(rank,world_size)  #GPU 

    with open("config.yaml","r") as f:
        config = yaml.safe_load(f)

    with open("generate_text_config.yaml","r") as f:
        generate_text_config = yaml.safe_load(f)

    if rank == 0 : 
        wandb.init(
            project="Foundation_models",
            name=args.run_name,
            config=config
        )



    train_dl, val_dl = fetch_train_val_dl()
    model = GPT2Model(config).to(rank)
    model = DDP(model,device_ids = [rank])


    # Check if a best model checkpoint exists and load it
    # best_model_path = "best_model_train_loss.pth"
    # if os.path.exists(best_model_path):
    #     model.load_state_dict(torch.load(best_model_path,weights_only=True, map_location=config.get("device", "cpu")))
    #     logging.info(f"Loaded best model from {best_model_path}")
    # else:
    #     logging.info("No best model checkpoint found. Training from scratch.")


    # model = torch.compile(model)

    if rank == 0 :
        num_parameters = sum(p.numel() for p in model.parameters())
        logging.info(f"Number of parameters: {num_parameters:,}")

    optimizer = torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],betas = (0.9,0.95),eps=1e-8)


    trainer = Trainer(
        model,
        train_dl,
        val_dl,
        loss_fn=cross_entropy,
        accuracy_fn=accuracy,
        optimizer=optimizer,
        config=config,
        rank = rank, 
        generate_text_config=generate_text_config,
        overfit_single_batch= False
    )

    trainer.train()

    if rank == 0 : 
        wandb.finish()
        torch.save(model.state_dict(), 'final_model.pth')

    destroy_process_group()
    


if __name__ == "__main__":
    if 'WORLD_SIZE' in os.environ : 
        world_size = int(os.environ["WORLD_SIZE"])
    else : 
        world_size = 1 

    if "LOCAL_RANK" in os.environ : 
        rank = int(os.environ['LOCAL_RANK'])
    elif 'RANK' in os.environ : 
        rank = int(os.environ['RANK'])
    else : 
        rank = 0 



    main(rank,world_size)


    
    
