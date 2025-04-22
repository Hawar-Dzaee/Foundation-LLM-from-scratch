import sys 
sys.path.append("..")


import tiktoken
from data.dataset import Data
from data.dataloader import get_data_loader


with open("/Users/hawardzaee/Desktop/Galaxy/MyLabV2/Foundation-LLM-from-scratch/L2_Advanced/GPT2/data/the-verdict.txt","r") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")


dataset = Data(
    raw_text=raw_text,
    tokenizer=tokenizer,
    context_length=2,
    stride=2
)

data_dl = get_data_loader(
    dataset=dataset,
    batch_size=10,
    shuffle=False,
    drop_last=True,
    num_workers=0)  

for x,y in data_dl:
    print(x)
    print(y)
    break