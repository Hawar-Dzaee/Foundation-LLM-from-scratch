import torch 
from torch import nn 

class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['embed_dim'])
        self.mha = nn.MultiheadAttention(
            embed_dim= config['embed_dim'],
            num_heads= config['num_heads'],
            batch_first= config['batch_first'],
            bias= config['Q_K_V_bias'],
            add_bias_kv= config['kv_bias'],
            dropout= config['dropout'],
            device= config['device']
            )
        self.dp1 = nn.Dropout(config['dropout'])

        self.ln2 = nn.LayerNorm(config['embed_dim'])
        self.mlp = nn.Sequential(
            nn.Linear(config['embed_dim'],2*config['embed_dim']),
            nn.GELU(),
            nn.Linear(2*config['embed_dim'],config['embed_dim']),
        )
        self.dp2 = nn.Dropout(config['dropout'])


    def forward(self,x):
       _,num_tokens,_ = x.shape

       shortcut = x
       x = self.ln1(x)
       x,_ = self.mha(
           x,x,x,
           attn_mask= torch.triu(torch.ones(num_tokens,num_tokens),diagonal=1).bool())
       x = self.dp1(x)

       x = x + shortcut

       shortcut = x
       x = self.ln2(x)
       x = self.mlp(x)
       x = self.dp2(x)
       x = x + shortcut
       return x
       

