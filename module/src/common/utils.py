import tiktoken
import torch


def text_to_tokens(
        text,
        tokenizer=tiktoken.get_encoding('gpt2'),
        tensor=True,
        ):
    tokens = tokenizer.encode(text,allowed_special={'<|endoftext|>'})
    if tensor:
        return torch.tensor(tokens,dtype=torch.long).unsqueeze(0)
    else:
        return tokens


def tokens_to_text(
        tokens,
        tokenizer=tiktoken.get_encoding('gpt2'),
        ):
    
    if type(tokens) is not torch.Tensor:
        tokens = torch.tensor(tokens)

    if len(tokens.shape) == 2 and tokens.shape[0] == 1:
        tokens = tokens.squeeze(0)

    # when we have -100 tokens in the output, we need to remove them
    if any(tokens < 0):
        tokens = tokens[:torch.argmin(tokens)]

    tokens = tokens.squeeze(0).tolist() # untensored
    text = tokenizer.decode(tokens)
    return text

