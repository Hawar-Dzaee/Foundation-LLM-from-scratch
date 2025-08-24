import torch 


print(torch.cuda.is_available())

tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])

tensor_1 = tensor_1.to('cuda')
tensor_2 = tensor_2.to('cuda')

print(tensor_1+tensor_2)