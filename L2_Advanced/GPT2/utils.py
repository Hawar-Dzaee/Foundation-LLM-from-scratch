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
    tokens = tokens.squeeze(0).tolist() # untensored
    text = tokenizer.decode(tokens)
    return text




def generate_text(
        text_to_generate,
        model,
        device,
        look_back=256,
        num_tokens_to_generate=20,
):
    starting_tokens = text_to_tokens(text_to_generate).to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(num_tokens_to_generate):
            starting_tokens = starting_tokens[:, -look_back:]
            logits = model(starting_tokens)[:,-1,:] #logits of the last token
            next_token_prob_distribution = torch.softmax(logits,dim=-1) 
            token_predicted = torch.argmax(next_token_prob_distribution,dim=-1,keepdim=True) 
            tokens = torch.concat([starting_tokens,token_predicted],dim=1) 
            starting_tokens = tokens
            text = tokens_to_text(tokens)
            text = text.replace("\n", " ")
            
    return text