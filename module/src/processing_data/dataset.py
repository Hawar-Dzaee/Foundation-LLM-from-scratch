from typing import List,Dict,Any

import torch
from torch.utils.data import Dataset
import pandas as pd
import tiktoken


class Data(Dataset):
    def __init__(
            self,
            raw_text:str,
            tokenizer:tiktoken,
            context_length:int,
            stride:int
            ):
        self.token_id = tokenizer.encode(raw_text,allowed_special={'<|endoftext|>'})
        self.num_tokens = len(self.token_id)
        self.X = []
        self.y = []

        assert self.num_tokens > context_length, "Number of tokens is less than Context length"

        for i in range(0,self.num_tokens-context_length,stride):
            X = self.token_id[i:i+context_length]
            y = self.token_id[i+1:i+context_length+1]
            self.X.append(torch.tensor(X,dtype=torch.long))
            self.y.append(torch.tensor(y,dtype=torch.long))
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]
            


class ClassificationDataset(Dataset):
    def __init__(self,csv_path,tokenizer,max_len=None,pad_token_id=50256):
        self.df = pd.read_csv(csv_path)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.df['message']
        ]

        if max_len is None : 
            # self.max_len = max([len(text) for text in self.encoded_texts])
            self.max_len = self._longest_max_len()
        else : 
            self.max_len = max_len
            self.encoded_texts = [
                encoded_text[:self.max_len]
                for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_len - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self,idx):
        encoded = self.encoded_texts[idx]
        label = self.df.iloc[idx]['label']
        return (
            torch.tensor(encoded,dtype=torch.long),
            torch.tensor(label,dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.df)

    def _longest_max_len(self):
        max_len = 0
        for encoded_text in self.encoded_texts:
            if len(encoded_text) > max_len:
                max_len = len(encoded_text)
        return max_len
        


class InstructionDataset(Dataset):
    def __init__(self,data:List[Dict[str,Any]],tokenizer:Any):
        self.data = data 

        self.encoded_texts = []

        for entry in self.data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Responsive:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.encoded_texts[idx]
    

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text
