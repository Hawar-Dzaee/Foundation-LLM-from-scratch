import torch 
from torch import nn 


class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['embed_dim'])
        self.mha = MultiheadAttention(
            context_window = config['context_window'],
            embed_dim= config['embed_dim'],
            num_heads= config['num_heads'],
            # batch_first= config['batch_first'],
            # bias= config['Q_K_V_bias'],
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
       x,_ = self.mha(x)
        #    attn_mask= torch.triu(torch.ones(num_tokens,num_tokens,device=x.device),diagonal=1).bool())
       x = self.dp1(x)

       x = x + shortcut

       shortcut = x
       x = self.ln2(x)
       x = self.mlp(x)
       x = self.dp2(x)
       x = x + shortcut
       return x
       

class MultiheadAttention(nn.Module):
  def __init__(
    self,
    context_window,
    embed_dim,
    num_heads,
    add_bias_kv=False,
    dropout=0,
    device=None
    ):
    super().__init__()

    # we will assume d_in == d_out and they are both embed_dim.

    # Handling dimensions
    assert embed_dim % num_heads == 0, 'Embedding must be divisible by Number of heads'
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = self.embed_dim//self.num_heads

    # W_q, W_k, W_v
    self.W_q      = nn.Linear(embed_dim,embed_dim,bias=False)
    self.W_k      = nn.Linear(embed_dim,embed_dim,bias=False)
    self.W_v      = nn.Linear(embed_dim,embed_dim,bias=False)
    self.out_proj = nn.Linear(embed_dim,embed_dim,bias=False)



    # Miscellaneous
    self.register_buffer('mask',torch.triu(torch.ones(context_window,context_window),diagonal=1))
    self.dropout = nn.Dropout(dropout)

    
  
  def forward(self,x):

    B_q,num_token_q,embed_dim_q = x.shape
    B_k,num_token_k,embed_dim_k = x.shape
    B_v,num_token_v,embed_dim_v = x.shape 



    Q = self.W_q(x)
    K = self.W_k(x)
    V = self.W_v(x)

    # splitting 
    Q = Q.view(B_q,num_token_q,self.num_heads,self.head_dim).transpose(1,2)
    K = K.view(B_k,num_token_k,self.num_heads,self.head_dim).transpose(1,2)
    V = V.view(B_v,num_token_v,self.num_heads,self.head_dim).transpose(1,2)

    # QK,mask,softmax,dropout
    attn_score = Q @ K.transpose(2,3)
    attn_score.masked_fill_(self.mask.bool()[:num_token_q,:num_token_k],-torch.inf)
    attn_weight = torch.softmax(attn_score/K.shape[-1]**0.5,dim=-1)
    attn_weight = self.dropout(attn_weight)

    # context_vec
    context_vec = attn_weight @ V

    # Putting the heads back together 
    context_vec = context_vec.transpose(1,2).contiguous().view(B_q,num_token_q,self.embed_dim)    # it doesn't matter which (B) you choose

    # projection 
    context_vec = self.out_proj(context_vec)

    return context_vec,attn_weight