import logging
import wandb
import torch

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(
            self,
            model,
            test_dl,
            loss_fn,
            accuracy_fn,
            device,
    ):

        self.model = model
        self.test_dl = test_dl
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.device = device


    def _run_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets)
            acc = self.accuracy_fn(logits, targets)
            return loss.item(), acc.item()


    
    def _log_metrics(self,test_loss,test_acc):
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc
        })
    
    def evaluate(self):
        test_loss,test_acc = 0,0

        for batch in self.test_dl:
            loss,acc = self._run_batch(batch)
            test_loss += loss
            test_acc += acc

        test_loss /= len(self.test_dl)
        test_acc /= len(self.test_dl)

        
        logger.info(f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")
        self._log_metrics(test_loss,test_acc)

        return test_loss,test_acc
