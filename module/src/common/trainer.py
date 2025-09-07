import time 
import logging
import wandb
import torch
from tqdm import tqdm
from common.inference import TextGeneration


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)



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
        overfit_single_batch=False
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
        self.seen_tokens = 0
        self.global_step = 0

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.log_ever_n_batches = config['log_ever_n_batches']


    def _run_batch_train(self, batch):
        self.model.train()
        self.model = self.model.to(self.device)
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        logits = self.model(inputs)
        loss = self.loss_fn(logits, targets)
        acc = self.accuracy_fn(logits,targets) #accuracy of the batch
        self.seen_tokens += inputs.numel() # diff 1 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(),acc.item()
    
    def _run_batch_val(self, batch):
        self.model.eval()
        with torch.no_grad():
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets)
            acc = self.accuracy_fn(logits,targets) 
            return loss.item(),acc.item()

    
    def _run_epoch_train(self):
        num_train_batches = len(self.train_dl)
        train_loss = 0
        train_acc = 0

        best_train_loss = float('inf')
        for batch_idx,batch in enumerate(self.train_dl):
            loss,acc = self._run_batch_train(batch)
            train_loss += loss
            train_acc += acc
            self.global_step += 1 

            # Step level- logging 
            if (batch_idx + 1) % self.log_ever_n_batches == 0:
                logging.info(
                    f"Batch {batch_idx+1:04d}/{num_train_batches} | "
                    f"Train Batch loss: {loss:.4f} | Train Batch acc: {acc:.4f}"
                    )
                wandb.log({
                    "train/loss_step": round(loss,4),
                    "train/acc_step": round(acc,4),
                    "train/seen tokens": self.seen_tokens,
                    "global_step": self.global_step
                })

                if loss < best_train_loss:
                    best_train_loss = loss
                    torch.save(self.model.state_dict(), f'best_model_train_loss.pth')
                    logging.info(f"New best model saved! Train loss: {loss:.4f}")

        train_loss /= num_train_batches
        train_acc /= num_train_batches

        return train_loss,train_acc


    def _run_epoch_val(self):
        num_val_batches = len(self.val_dl)
        val_loss = 0
        val_acc = 0

        best_val_loss = float('inf')
        best_val_acc = 0
        for batch_idx,batch in enumerate(self.val_dl):
            loss,acc = self._run_batch_val(batch)
            val_loss += loss
            val_acc += acc

            if loss < best_val_loss:
                best_val_loss = loss
                torch.save(self.model.state_dict(), f'best_model_val_loss.pth')
                logging.info(f"New best model saved! Val loss: {loss:.4f}")

        val_loss /= num_val_batches
        val_acc /= num_val_batches
        
        return val_loss,val_acc
    
    
    def _log_metrics_epoch(self,train_loss,val_loss,train_acc,val_acc,seen_tokens):
        """Log aggregated metrics at the end of each epoch."""
        wandb.log({
            "train/loss_epoch": round(train_loss,4),
            "train/acc_epoch": round(train_acc,4),
            "val/loss_epoch": round(val_loss,4),
            "val/acc_epoch": round(val_acc,4),
            "global_step": self.global_step
        })



    def train(self):
        start_time = time.time()
        
        # Create a single progress bar for epochs
        epoch_pbar = tqdm(range(self.config["epochs"]), desc="Training Epochs", unit="epoch")
        
        for epoch in epoch_pbar:
            logging.info(f"Epoch {epoch+1}/{self.config['epochs']} - Training ...")

            train_loss,train_acc = self._run_epoch_train()
            val_loss,val_acc = self._run_epoch_val()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Epoch Level Logging 
            self._log_metrics_epoch(train_loss,val_loss,train_acc,val_acc,self.seen_tokens)
            
            # Update progress bar with current metrics
            epoch_pbar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "train_acc": f"{train_acc:.4f}",
                "val_acc": f"{val_acc:.4f}"
            })

            
            logging.info(
                f"Epoch {epoch+1}/{self.config['epochs']} | "
                f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f}  | Val acc: {val_acc:.4f}"
                )

            # Sample Text Generation
            if self.generate_text_config["input_text"] :
                text_generation = TextGeneration(
                    model = self.model,
                    top_k= self.generate_text_config["top_k"],
                    temperature= self.generate_text_config["temperature"],
                    look_back= self.generate_text_config["look_back"],
                    num_tokens_to_generate= self.generate_text_config["num_tokens_to_generate"],
                    device= self.generate_text_config['device'],
                    )
                input_text,output_text = text_generation.chat(
                    input_text= self.generate_text_config["input_text"],
                )
                

                logging.info(f"Input Text: {input_text}\nOutput Text: {output_text}")
                wandb.log({
                    "samples/input_text": input_text,
                    "samples/output_text": output_text,
                    "global_step": self.global_step,
                })

        duration = time.time() - start_time
        logging.info(f'Total Training Duration : {duration:.4f}')

        return self.history

