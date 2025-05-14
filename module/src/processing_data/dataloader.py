import torch
from torch.utils.data import DataLoader

def get_data_loader(dataset,batch_size,shuffle,drop_last,num_workers=0,collate_fn=None): 
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn
        )


def instruction_collate_fn(
      batch,    
      pad_token_id=50256,
      ignore_index = -100,
      allowed_max_length = None,
      device='cpu'
      ):

  max_batch_length  = max(len(i) for i in batch)
  padded_input_batch,padded_output_batch = [],[]

  for each_sample in batch :
    # figuring out padding
    padd_added_to_input  = [pad_token_id]*(max_batch_length-len(each_sample))
    
    padd_added_to_output = padd_added_to_input + [pad_token_id] # add one more pad token to the output(y)
    padd_added_to_output[1:] = [ignore_index]* len(padd_added_to_output[1:]) # after the first, ignore the rest.

    # input/outputs
    inputs = each_sample + padd_added_to_input
    outputs = each_sample[1:] + padd_added_to_output

    if allowed_max_length is not None : 
      inputs = inputs[:allowed_max_length]
      outputs = outputs[:allowed_max_length]

    # to tensors
    inputs = torch.tensor(inputs)
    outputs= torch.tensor(outputs)

    padded_input_batch.append(inputs)
    padded_output_batch.append(outputs)


  tensor_input_batch = torch.stack(padded_input_batch,dim=0).to(device)
  tensor_output_batch = torch.stack(padded_output_batch,dim=0).to(device)

  return  tensor_input_batch,tensor_output_batch