import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self,raw_text,tokenizer,context_length,stride):
        self.token_id = tokenizer.encode(raw_text,allowed_special={'<|endoftext|>'})
        self.X = []
        self.y = []

        for i in range(0,len(self.token_id)-context_length,stride):
            X = self.token_id[i:i+context_length]
            y = self.token_id[i+1:i+context_length+1]
            self.X.append(torch.tensor(X,dtype=torch.long))
            self.y.append(torch.tensor(y,dtype=torch.long))
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]
            


