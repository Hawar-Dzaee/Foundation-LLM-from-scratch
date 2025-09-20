import time 
import logging
import wandb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from common.inference import TextGeneration
import os
import platform


logger = logging.getLogger(__name__)


def ddp_setup(rank, world_size):
    """
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    # Only set MASTER_ADDR and MASTER_PORT if not already defined by torchrun
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"

    # initialize process group
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows isn't built with support
        os.environ["USE_LIBUV"] = "0"
        # Windows users may have to use "gloo" instead of "nccl" as backend
        # gloo: Facebook Collective Communication Library
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


def ddp_cleanup():
    """Clean up the process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


class Trainer:
    def __init__(
        self,
        model,
        train_dl,
        val_dl,
        loss_fn,
        accuracy_fn,
        optimizer,
        config,
        generate_text_config,
        overfit_single_batch=False,
        rank=None,  # Add rank parameter for DDP
        world_size=None  # Add world_size parameter for DDP
    ):

        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.optimizer = optimizer
        self.config = config
        self.device = config['device']
        self.generate_text_config = generate_text_config
        self.overfit_single_batch = overfit_single_batch
        
        # DDP-related attributes
        self.rank = rank
        self.world_size = world_size
        self.is_ddp = rank is not None and world_size is not None
        self.is_main_process = not self.is_ddp or rank == 0

        self.seen_tokens = 0
        self.global_step = 0

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.log_ever_n_batches = config['log_ever_n_batches']

        # Initialize DDP if running in distributed mode
        if self.is_ddp:
            ddp_setup(rank, world_size)
            self.model = self.model.to(self.device)
            self.model = DDP(self.model, device_ids=[rank])
            
            # Only initialize wandb on main process
            if self.is_main_process and wandb.run is None:
                # wandb.init() should be called here if not already done
                pass
        else:
            self.model = self.model.to(self.device)

    def _run_batch_train(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets)
            
        acc = self.accuracy_fn(logits, targets) 
        self.seen_tokens += inputs.numel()
        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item(), acc.item()
    
    def _run_batch_val(self, batch):
        self.model.eval()
        with torch.no_grad():
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets)
            acc = self.accuracy_fn(logits, targets) 
            return loss.item(), acc.item()

    def _run_epoch_train(self):
        num_train_batches = len(self.train_dl)
        train_loss = 0
        train_acc = 0

        best_train_loss = float('inf')
        
        # Set epoch for DistributedSampler if using DDP
        if self.is_ddp and hasattr(self.train_dl.sampler, 'set_epoch'):
            self.train_dl.sampler.set_epoch(self.current_epoch)
        
        for batch_idx, batch in enumerate(self.train_dl):
            start_time = time.time()
            loss, acc = self._run_batch_train(batch)
            train_loss += loss
            train_acc += acc
            self.global_step += 1 

            end_time = time.time()  
            if self.is_main_process:
                print(f"Batch duration: {(end_time-start_time)*1000:.3f} ms")

            # Step level logging - only on main process
            if self.is_main_process and (batch_idx + 1) % self.log_ever_n_batches == 0:
                logging.info(
                    f"Batch {batch_idx+1:04d}/{num_train_batches} | "
                    f"Train Batch loss: {loss:.4f} | Train Batch acc: {acc:.4f}"
                )
                if wandb.run:
                    wandb.log({
                        "train/loss_step": round(loss, 4),
                        "train/acc_step": round(acc, 4),
                        "train/seen tokens": self.seen_tokens,
                        "global_step": self.global_step
                    })

                if loss < best_train_loss:
                    best_train_loss = loss
                    # Save model state dict properly for DDP
                    model_to_save = self.model.module if self.is_ddp else self.model
                    torch.save(model_to_save.state_dict(), f'best_model_train_loss.pth')
                    logging.info(f"New best model saved! Train loss: {loss:.4f}")

            if self.overfit_single_batch:
                break

        # Synchronize metrics across all processes if using DDP
        if self.is_ddp:
            train_loss_tensor = torch.tensor(train_loss, device=self.device)
            train_acc_tensor = torch.tensor(train_acc, device=self.device)
            
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
            
            train_loss = train_loss_tensor.item() / self.world_size
            train_acc = train_acc_tensor.item() / self.world_size

        train_loss /= num_train_batches
        train_acc /= num_train_batches

        return train_loss, train_acc

    def _run_epoch_val(self):
        num_val_batches = len(self.val_dl)
        val_loss = 0
        val_acc = 0

        best_val_loss = float('inf')
        for batch_idx, batch in enumerate(self.val_dl):
            loss, acc = self._run_batch_val(batch)
            val_loss += loss
            val_acc += acc

            if self.is_main_process and loss < best_val_loss:
                best_val_loss = loss
                model_to_save = self.model.module if self.is_ddp else self.model
                torch.save(model_to_save.state_dict(), f'best_model_val_loss.pth')
                logging.info(f"New best model saved! Val loss: {loss:.4f}")

            if self.overfit_single_batch:
                break

        # Synchronize validation metrics across all processes if using DDP
        if self.is_ddp:
            val_loss_tensor = torch.tensor(val_loss, device=self.device)
            val_acc_tensor = torch.tensor(val_acc, device=self.device)
            
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.SUM)
            
            val_loss = val_loss_tensor.item() / self.world_size
            val_acc = val_acc_tensor.item() / self.world_size

        val_loss /= num_val_batches
        val_acc /= num_val_batches
        
        return val_loss, val_acc
    
    def _log_metrics_epoch(self, train_loss, val_loss, train_acc, val_acc, seen_tokens):
        """Log aggregated metrics at the end of each epoch."""
        if self.is_main_process and wandb.run:
            wandb.log({
                "train/loss_epoch": round(train_loss, 4),
                "train/acc_epoch": round(train_acc, 4),
                "val/loss_epoch": round(val_loss, 4),
                "val/acc_epoch": round(val_acc, 4),
                "global_step": self.global_step
            })

    def train(self):
        start_time = time.time()
        
        # Only show progress bar on main process
        if self.is_main_process:
            epoch_pbar = tqdm(range(self.config["epochs"]), desc="Training Epochs", unit="epoch")
        else:
            epoch_pbar = range(self.config["epochs"])
        
        for epoch in epoch_pbar:
            self.current_epoch = epoch  # Store current epoch for DistributedSampler
            
            torch.cuda.synchronize()
            epoch_start_time = time.time()
            
            if self.is_main_process:
                logging.info(f"Epoch {epoch+1}/{self.config['epochs']} - Training ...")

            train_loss, train_acc = self._run_epoch_train()
            val_loss, val_acc = self._run_epoch_val()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Epoch Level Logging - only on main process
            if self.is_main_process:
                self._log_metrics_epoch(train_loss, val_loss, train_acc, val_acc, self.seen_tokens)
                
                # Update progress bar with current metrics
                if hasattr(epoch_pbar, 'set_postfix'):
                    epoch_pbar.set_postfix({
                        "train_loss": f"{train_loss:.4f}",
                        "val_loss": f"{val_loss:.4f}",
                        "train_acc": f"{train_acc:.4f}",
                        "val_acc": f"{val_acc:.4f}"
                    })

            torch.cuda.synchronize()
            epoch_duration = time.time() - epoch_start_time
            formatted_epoch_time = time.strftime("%H:%M:%S", time.gmtime(epoch_duration))

            if self.is_main_process:
                logging.info(
                    f"Epoch {epoch+1}/{self.config['epochs']} | "
                    f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
                    f"Val loss: {val_loss:.4f}  | Val acc: {val_acc:.4f} | "
                    f"Epoch time: {formatted_epoch_time} ({epoch_duration:.2f} sec)"
                )

                if wandb.run:
                    wandb.log({
                        "epoch_time_seconds": epoch_duration,
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "global_step": self.global_step,
                    })

            # Sample Text Generation - only on main process
            if self.is_main_process and self.generate_text_config["input_text"]:
                # Use the underlying model for text generation, not the DDP wrapper
                model_for_generation = self.model.module if self.is_ddp else self.model
                
                text_generation = TextGeneration(
                    model=model_for_generation,
                    top_k=self.generate_text_config["top_k"],
                    temperature=self.generate_text_config["temperature"],
                    look_back=self.generate_text_config["look_back"],
                    num_tokens_to_generate=self.generate_text_config["num_tokens_to_generate"],
                    device=self.generate_text_config['device'],
                )
                input_text, output_text = text_generation.chat(
                    input_text=self.generate_text_config["input_text"],
                )

                logging.info(f"Input Text: {input_text}\nOutput Text: {output_text}")
                if wandb.run:
                    wandb.log({
                        "samples/input_text": input_text,
                        "samples/output_text": output_text,
                        "global_step": self.global_step,
                    })

            if self.is_main_process:
                logging.info("="*100)

        duration = time.time() - start_time
        formatted_total_time = time.strftime("%H:%M:%S", time.gmtime(duration))
        
        if self.is_main_process:
            logging.info(f"Total Training Duration: {formatted_total_time} ({duration:.2f} sec)")
            if wandb.run:
                wandb.log({"total_training_time_seconds": duration})

        # Clean up DDP
        if self.is_ddp:
            ddp_cleanup()

        return self.history


# Example usage function for DDP training
def main_ddp(rank, world_size, config, model, train_dataset, val_dataset, loss_fn, accuracy_fn, optimizer, generate_text_config):
    """
    Main function for DDP training
    """
    # Create DistributedSamplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Create DataLoaders with DistributedSamplers
    train_dl = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        sampler=train_sampler,
        num_workers=config.get('num_workers', 0),
        pin_memory=True
    )
    
    val_dl = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        sampler=val_sampler,
        num_workers=config.get('num_workers', 0),
        pin_memory=True
    )
    
    # Initialize trainer with DDP parameters
    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        optimizer=optimizer,
        config=config,
        generate_text_config=generate_text_config,
        rank=rank,
        world_size=world_size
    )
    
    # Start training
    history = trainer.train()
    return history