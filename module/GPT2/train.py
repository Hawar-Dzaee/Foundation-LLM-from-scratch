import logging
import wandb

from evaluation import eval
from utils import generate_text


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)



class Trainer:
    def __init__(self,model,train_loader,val_loader,loss_fn,optimizer,num_epochs,
                 device,text_to_generate=None,look_back=256,num_tokens_to_generate=20):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def _run_batch(self,batch):
        inputs,targets = batch
        inputs,targets = inputs.to(self.device),targets.to(self.device)
        logits = self.model(inputs)
        loss = self.loss_fn(logits,targets)
        return loss

def traininng_loop(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    num_epochs,
    device,
    text_to_generate = None,
    look_back = 256,
    num_tokens_to_generate = 20,
):
    seen_tokens = 0

    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = []
        
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            seen_tokens += inputs.numel()
        train_loss = sum(train_loss)/len(train_loss)
            
        logging.info(f"Seen tokens: {seen_tokens}")
        logging.info(f"Loss: {train_loss:.4f}")

        wandb.log({
            "loss": train_loss,
            "seen_tokens": seen_tokens
        })



        eval(model, val_loader, loss_fn, device)

        if text_to_generate:
            generated_text = generate_text(
                text_to_generate,
                model,
                device,
                look_back,
                num_tokens_to_generate
                )
            logging.info(f"Generated text: {generated_text}")


        logging.info("="*50)
