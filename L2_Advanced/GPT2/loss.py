from torch import nn 


def loss_fn(logits,targets,device):
    logits,targets = logits.to(device),targets.to(device)
    return nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten())
