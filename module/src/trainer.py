import time 
import logging
import wandb
import torch
from tqdm import tqdm
from inference import TextGeneration


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
        device,
        generate_text_config
    ):

        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.generate_text_config = generate_text_config
        self.seen_tokens = 0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }


    def _run_batch_train(self, batch):
        self.model.train()
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
            acc = self.accuracy_fn(logits,targets) #accuracy of the batch
            return loss.item(),acc.item()

    
    def _run_epoch(self):
        train_loss,val_loss = 0,0
        train_acc,val_acc = 0,0
        
        train_iter = tqdm(self.train_dl, desc="Training", leave=False)
        for batch in train_iter:
            loss,acc = self._run_batch_train(batch)
            train_loss += loss
            train_acc += acc

        train_loss /= len(self.train_dl)
        train_acc /= len(self.train_dl)

        val_iter = tqdm(self.val_dl, desc="Validation", leave=False)
        for batch in val_iter:
            loss,acc = self._run_batch_val(batch)
            val_loss += loss
            val_acc += acc
            
        val_loss /= len(self.val_dl)
        val_acc /= len(self.val_dl)
        
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
        for epoch in range(self.config["epochs"]):
            logging.info(f"Epoch {epoch+1}/{self.config['epochs']} - Training ...")
            train_loss,val_loss,train_acc,val_acc = self._run_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self._log_metrics(train_loss,val_loss,train_acc,val_acc,self.seen_tokens)
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

