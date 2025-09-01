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
        generate_text_config
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
        self.seen_tokens = 0
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

        # print(f"Model device: {next(self.model.parameters()).device}")
        # print(f"Input device: {inputs.device}")
        # print(f"Input device: {targets.device}")

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
            acc = self.accuracy_fn(logits,targets) #accuracy of the batch
            return loss.item(),acc.item()

    
    def _run_epoch(self):
        num_train_batches = len(self.train_dl)
        num_val_batches = len(self.val_dl)
        train_loss,val_loss = 0,0
        train_acc,val_acc = 0,0 

        best_train_loss = float('inf')
        for batch_idx,batch in enumerate(self.train_dl):
            loss,acc = self._run_batch_train(batch)
            train_loss += loss
            train_acc += acc
            if (batch_idx + 1) % self.log_ever_n_batches == 0:
                logging.info(f"Batch {batch_idx+1:04d}/{num_train_batches} | Train Batch loss: {loss:.4f} | Train Batch acc: {acc:.4f}")
                if loss < best_train_loss:
                    best_train_loss = loss
                    torch.save(self.model.state_dict(), f'best_model_train_loss.pth')
                    logging.info(f"New best model saved! Train loss: {loss:.4f}")

        train_loss /= num_train_batches
        train_acc /= num_train_batches


        best_val_loss = float('inf')
        best_val_acc = 0
        for batch_idx,batch in enumerate(self.val_dl):
            loss,acc = self._run_batch_val(batch)
            val_loss += loss
            val_acc += acc
            if (batch_idx + 1) % self.log_ever_n_batches == 0:
                logging.info(f"Batch {batch_idx+1:04d}/{num_val_batches} | Val Batch loss: {loss:.4f} | Val Batch acc: {acc:.4f}")
                if loss < best_val_loss:
                    best_val_loss = loss
                    torch.save(self.model.state_dict(), f'best_model_val_loss.pth')
                    logging.info(f"New best model saved! Val loss: {loss:.4f}")

        val_loss /= num_val_batches
        val_acc /= num_val_batches
        
        return train_loss,val_loss,train_acc,val_acc
    
    
    def _log_metrics(self,train_loss,val_loss,train_acc,val_acc,seen_tokens):
        wandb.log({
            "train loss": round(train_loss,4),
            "val loss": round(val_loss,4),
            "train acc": round(train_acc,4),
            "val acc": round(val_acc,4),
            "seen tokens": seen_tokens # diff 2 
        })


    def train(self):
        start_time = time.time()
        
        # Create a single progress bar for epochs
        epoch_pbar = tqdm(range(self.config["epochs"]), desc="Training Epochs", unit="epoch")
        
        for epoch in epoch_pbar:
            logging.info(f"Epoch {epoch+1}/{self.config['epochs']} - Training ...")
            train_loss,val_loss,train_acc,val_acc = self._run_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self._log_metrics(train_loss,val_loss,train_acc,val_acc,self.seen_tokens)
            
            # Update progress bar with current metrics
            epoch_pbar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "train_acc": f"{train_acc:.4f}",
                "val_acc": f"{val_acc:.4f}"
            })

            
            logging.info(f"Epoch {epoch+1}/{self.config['epochs']} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f}  | Val acc: {val_acc:.4f}")

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

        duration = time.time() - start_time
        logging.info(f'Total Training Duration : {duration:.4f}')

        return self.history

