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

def generate_text(model,config):
        starting_tokens = text_to_tokens(config["text_to_generate"]).to(config['device'])

        model.eval()
        with torch.no_grad():
            for _ in range(config["num_tokens_to_generate"]):
                starting_tokens = starting_tokens[:, -config["look_back"]:]
                logits = model(starting_tokens)[:,-1,:] #logits of the last token

                if config["top_k"] : 
                    top_values, _ = torch.topk(logits,config["top_k"])
                    logits = torch.where(
                        logits<top_values[:,-1],
                        -torch.inf,
                        logits
                        )
                    logits = logits/(config["temperature"]+1e-7)
                    probs  = torch.softmax(logits,dim=-1) 
                    token_predicted = torch.multinomial(probs,num_samples=1)
                else : 
                    token_predicted = torch.argmax(logits,dim=-1,keepdim=True) 

                tokens = torch.concat([starting_tokens,token_predicted],dim=1) 
                text = tokens_to_text(tokens)
                text = text.replace("\n", " ")
                starting_tokens = tokens
        return text
