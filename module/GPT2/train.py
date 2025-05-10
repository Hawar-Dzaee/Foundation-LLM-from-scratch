import logging
import wandb
import torch

from evaluation import eval
from utils import text_to_tokens,tokens_to_text


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
        optimizer,
        config,
        device,
        generate_text_config
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.generate_text_config = generate_text_config
        self.train_loss = []
        self.val_loss = []
        self.seen_tokens = 0



    def _run_batch(self, batch,training):
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        logits = self.model(inputs)
        loss = self.loss_fn(logits, targets)

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.seen_tokens += inputs.numel()
        return loss.item()

    
    def _run_epoch(self):
        train_loss,val_loss = 0,0

        self.model.train()
        for batch in self.train_loader:
            loss = self._run_batch(batch,training=True)
            train_loss += loss

        train_loss = train_loss / len(self.train_loader)
        self.train_loss.append(train_loss)

        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self._run_batch(batch,training=False)
                val_loss += loss

        val_loss = val_loss / len(self.val_loader)
        self.val_loss.append(val_loss)
        return train_loss,val_loss
    
    def _log_metrics(self,epoch,train_loss,val_loss):
        wandb.log({
            "train loss": train_loss,
            "val loss": val_loss,
            "seen tokens": self.seen_tokens
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
            train_loss,val_loss = self._run_epoch()
            self._log_metrics(epoch,train_loss,val_loss)
            if generate_text:
                generated_text = self._generate_text()
                logging.info(f"Generated text: {generated_text}")

        return self.train_loss,self.val_loss

