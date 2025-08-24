import torch 
from torch import nn 

class Embeddings(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.token_embedding = nn.Embedding(config['vocab_size'],config['embed_dim'])
        self.positional_encoding = nn.Embedding(config['context_window'],config['embed_dim'])

    def forward(self,x):
        B,num_token = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.positional_encoding(torch.arange(num_token,device = x.device))
        return tok_emb + pos_emb