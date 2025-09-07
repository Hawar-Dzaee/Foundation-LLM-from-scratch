import os 
from typing import List,Dict,Any

import logging 

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import pandas as pd
import tiktoken


logger = logging.getLogger(__name__)

# Text corpus 
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

# Tiny story 
class TinyStoryData(Dataset):

    def __init__(
            self,
            dataset:str,
            tokenizer:tiktoken,
            cache_file : str,
            max_length:int = 512,
            ):
        self.tokenizer = tokenizer
        self.dataset = []

        if os.path.exists(cache_file):
            logging.info(f'Loading preprocessed Data from {cache_file}')
            self.dataset = torch.load(cache_file)


        else : 
            # get rid of samples that are longer than max_length
            for item in tqdm(dataset,desc="Processing Dataset"):
                text = item['text']
                tokenized = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'}) 
                tokenized += self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})

                if len(tokenized) <= max_length:
                    self.dataset.append(tokenized)

            logging.info(f"Saving processed data to {cache_file}")
            torch.save(self.dataset,f'{cache_file}')
            logging.info(f"Processed Data has been saved to {cache_file}")

            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):        
        return self.dataset[idx]



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
