import yaml
import chainlit 
import torch 
import tiktoken

from model_components.gpt2 import GPT2Model
from common.inference import TextGeneration


tokenizer = tiktoken.get_encoding("gpt2")


with open("config.yaml",'r') as f : 
    config = yaml.safe_load(f)

model = GPT2Model(config)
model_weights = torch.load("best_model_val_loss.pth")
model.load_state_dict(model_weights)


@chainlit.on_message
async def main(message:chainlit.Message):
    text_generation = TextGeneration(
        model = model, 
        device= "cuda" if torch.cuda.is_available() else "cpu",
        top_k= 4,
        temperature= 1.0,
        look_back= 100,
        num_tokens_to_generate=90,
    )

    _,output_text = text_generation.chat(
        input_text= message.content
    )

    await chainlit.Message(
        content=f'{output_text}'
    ).send()

