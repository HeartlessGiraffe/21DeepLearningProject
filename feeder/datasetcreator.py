from torch.utils.data import Dataset
class cifar10_dataset(Dataset):
    def __init__(self, data_set, labels):
        self.data_set = data_set
        self.labels = labels
        self.count = len(labels)
    
    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return (self.data_set[index], self.labels[index])


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def from_dict_to_tensor(data_dict):
    import torch
    import numpy as np 
    data = data_dict[b'data']
    labels = data_dict[b'labels']
    data = data.reshape(10000, 3, 32, 32).astype('float')/255
    data = (data - 0.5)*2
    data = torch.tensor(data)
    labels = torch.tensor(np.array(labels, dtype = 'int'))
    return data, labels
