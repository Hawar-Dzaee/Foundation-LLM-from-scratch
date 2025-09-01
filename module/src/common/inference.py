import torch

from .utils import text_to_tokens,tokens_to_text



class TextGeneration:
    def __init__(
            self,
            model,
            top_k = None, 
            temperature =1.0,
            look_back = 100,
            num_tokens_to_generate = 50,
            eos_token_id = 50256,
            device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.top_k = top_k
        self.temperature = temperature
        self.look_back = look_back
        self.num_tokens_to_generate = num_tokens_to_generate
        self.eos_token_id = eos_token_id
        self.device = device


    def chat(self,input_text):
        input_tokens = text_to_tokens(input_text).to(self.device)
        output_tokens = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(self.num_tokens_to_generate):
                input_tokens = input_tokens[:,-self.look_back:]
                logits = self.model(input_tokens)[:,-1,:]
                predicted_token = self._token_prediction(logits)
                if predicted_token.item() == self.eos_token_id : 
                    break  
                output_tokens.append(predicted_token.item())
                tokens = torch.cat([input_tokens,predicted_token],dim=1)
                input_tokens = tokens 

        output_text = tokens_to_text(output_tokens)
        

        return input_text, output_text




    def _token_prediction(self,logits):
        if self.top_k: 
            top_values, _ = torch.topk(logits,self.top_k)
            logits = torch.where(
                logits < top_values[:,-1],
                -torch.inf,
                logits
            )
            logits = logits/(self.temperature+1e-7)
            probs = torch.softmax(logits,dim=-1)
            predicted_token = torch.multinomial(probs,num_samples=1)
        else : 
            predicted_token = torch.argmax(logits,dim=-1,keepdim=True)
        
        return predicted_token