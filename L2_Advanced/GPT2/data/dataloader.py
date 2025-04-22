from torch.utils.data import DataLoader

def get_data_loader(dataset,batch_size,shuffle,drop_last,num_workers): 
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
        )