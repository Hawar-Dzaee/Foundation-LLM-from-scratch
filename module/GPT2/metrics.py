from torch import nn 

def  cross_entropy(logits, targets):
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(logits.flatten(0, 1), targets.flatten())

def classification_loss(logits, targets):
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(logits[:, -1, :], targets)


def accuracy(logits,targets):
    return (logits.flatten(0,1).argmax(dim=1) == targets.flatten()).float().mean()