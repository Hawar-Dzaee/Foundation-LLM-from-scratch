from torch import nn 

def  cross_entropy(logits, targets):
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(logits.flatten(0, 1), targets.flatten())