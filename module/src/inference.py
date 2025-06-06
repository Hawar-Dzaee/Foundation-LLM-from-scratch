import torch

from utils import text_to_tokens,tokens_to_text



def generate_text(
          model,
          device,
          top_k=None,
          temperature=1.0,
          look_back=10,
          num_tokens_to_generate=100,
          text_to_generate="",
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
                text = tokens_to_text(tokens)
                text = text.replace("\n", " ")
                starting_tokens = tokens
        return text