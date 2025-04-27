import torch.nn.functional as F

def loss_fn(logits, targets):
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())