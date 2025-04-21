from torch.utils.data import DataLoader
import tiktoken
from dataset import Data

def get_data_loader(
        raw_text:str,
        tokenizer:tiktoken.Encoding,
        context_length:int,
        stride:int,
        batch_size:int,
        shuffle:bool,
        drop_last:bool,
        num_workers:int
        ):
    dataset = Data(raw_text,tokenizer,context_length,stride)
    data_dl = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    return data_dl