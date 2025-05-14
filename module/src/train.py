import logging
import wandb
import torch
import numpy as np
from utils import text_to_tokens,tokens_to_text
from metrics import accuracy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)



class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        accuracy_fn,
        optimizer,
        config,
        device,
        generate_text_config
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.generate_text_config = generate_text_config
        self.train_losses,self.val_losses = [],[]
        self.train_accs,self.val_accs = [],[]
        self.seen_tokens = 0



    def _run_batch(self, batch,training):
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        logits = self.model(inputs)
        loss = self.loss_fn(logits, targets)
        acc = self.accuracy_fn(logits,targets) #accuracy of the batch

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.seen_tokens += inputs.numel()
        return loss.item(),acc.item()

    
    def _run_epoch(self):
        train_loss,val_loss = 0,0
        train_acc,val_acc = 0,0

        self.model.train()
        for batch in self.train_loader:
            loss,acc = self._run_batch(batch,training=True)
            train_loss += loss
            train_acc += acc

        train_loss = train_loss/len(self.train_loader)
        train_acc = train_acc/len(self.train_loader)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)

        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                loss,acc = self._run_batch(batch,training=False)
                val_loss += loss
                val_acc += acc

        val_loss = val_loss/len(self.val_loader)
        val_acc = val_acc/len(self.val_loader)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        
        return train_loss,val_loss,train_acc,val_acc
    
    def _log_metrics(self,train_loss,val_loss,train_acc,val_acc,seen_tokens):
        wandb.log({
            "train loss": round(train_loss,4),
            "val loss": round(val_loss,4),
            "train acc": round(train_acc,4),
            "val acc": round(val_acc,4),
            "seen tokens": seen_tokens
        })

    def _generate_text(self):
        starting_tokens = text_to_tokens(self.generate_text_config["text_to_generate"]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for _ in range(self.generate_text_config["num_tokens_to_generate"]):
                starting_tokens = starting_tokens[:, -self.generate_text_config["look_back"]:]
                logits = self.model(starting_tokens)[:,-1,:] #logits of the last token

                if self.generate_text_config["top_k"] : 
                    top_values, _ = torch.topk(logits,self.generate_text_config["top_k"])
                    logits = torch.where(
                        logits<top_values[:,-1],
                        -torch.inf,
                        logits
                        )
                    logits = logits/(self.generate_text_config["temperature"]+1e-7)
                    probs  = torch.softmax(logits,dim=-1) 
                    token_predicted = torch.multinomial(probs,num_samples=1)
                else : 
                    token_predicted = torch.argmax(logits,dim=-1,keepdim=True) 

                tokens = torch.concat([starting_tokens,token_predicted],dim=1) 
                text = tokens_to_text(tokens)
                text = text.replace("\n", " ")
                starting_tokens = tokens

                
        return text


    def train(self, epochs,generate_text=False):
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch+1}/{epochs}")
            train_loss,val_loss,train_acc,val_acc = self._run_epoch()
            self._log_metrics(train_loss,val_loss,train_acc,val_acc,self.seen_tokens)
            logging.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")

            if generate_text:
                generated_text = self._generate_text()
                logging.info(f"Generated text: {generated_text}")

        return self.train_losses,self.val_losses,self.train_accs,self.val_accs

