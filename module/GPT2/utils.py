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

    # when we have -100 tokens in the output, we need to remove them
    if any(tokens < 0):
        tokens = tokens[:torch.argmin(tokens)]

    tokens = tokens.squeeze(0).tolist() # untensored
    text = tokenizer.decode(tokens)
    return text




def generate_text(
        text_to_generate,
        model,
        device,
        look_back=256,
        num_tokens_to_generate=20,
        temperature = 1.0, 
        top_k = None,
        eos_id = None 
):
    starting_tokens = text_to_tokens(text_to_generate).to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(num_tokens_to_generate):
            starting_tokens = starting_tokens[:, -look_back:]
            logits = model(starting_tokens)[:,-1,:] #logits of the last token

            if top_k : 
                top_values, _ = torch.topk(logits,top_k)
                logits = torch.where(
                    logits<top_values[:,-1],
                    -torch.inf,
                    logits
                    )
                logits = logits/(temperature+1e-7)
                probs  = torch.softmax(logits,dim=-1) 
                token_predicted = torch.multinomial(probs,num_samples=1)
            else : 
                token_predicted = torch.argmax(logits,dim=-1,keepdim=True) 

            tokens = torch.concat([starting_tokens,token_predicted],dim=1) 
            starting_tokens = tokens
            text = tokens_to_text(tokens)
            text = text.replace("\n", " ")
            
    return text