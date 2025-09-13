In story dataset

- "train" split is used for both train & validation.
- "validation" is used for test. 

the evalute_gpt2.py saves ...

1- "best_model_train_loss.pth" Logic :  @ every 30th batch if we have better loss we will save it 
2- "best_model_val_loss.pth"   Logic :  @ every 30th batch if we have better loss we will save it 
3- by the end of training it will save "final_model.pth"



# Speed up 

- torch.set_float32_matmul_precision("high")  (at the top of your script )
- with torch.autocast(device="cuda",dtype=torch.bfloat16) that contexts logits and loss 
- you should be using torch.compile(model) by default (unless you are debugging otherwise there is no reason not to use it)