import logging
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def eval(
    model,
    val_loader,
    loss_fn,
    device,
):
    model.eval()
    model.to(device)
    val_loss = []
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = loss_fn(logits, targets)

            val_loss.append(loss.item())
            
    val_loss = sum(val_loss)/len(val_loss)
    logging.info(f"Validation Loss: {val_loss:.4f}")
            
            
            