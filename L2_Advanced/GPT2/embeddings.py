import torch 
from torch import nn 

class Embeddings(nn.Module):
    def __init__(self,vocab_size,embedding_dim,context_window):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(vocab_size,embedding_dim)
        self.positional_encoding = nn.Embedding(context_window,embedding_dim)

    def forward(self,x):
        B,num_token = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.positional_encoding(torch.arange(num_token))
        return tok_emb + pos_emb