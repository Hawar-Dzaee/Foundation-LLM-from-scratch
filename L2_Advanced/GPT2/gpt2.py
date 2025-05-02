from typing import Dict, Any
import torch 
from torch import nn 
from embeddings import Embeddings
from transformer_block import TransformerBlock



class GPT2Model(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.embeddings = Embeddings(config)
        self.dropout = nn.Dropout(config['dropout'])
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config['n_layers'])])
        self.ln = nn.LayerNorm(config['embed_dim'])
        self.head = nn.Linear(config['embed_dim'],config['num_classes'],bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits