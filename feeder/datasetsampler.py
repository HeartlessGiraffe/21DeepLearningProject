from .datasetcreator import unpickle
from .datasetcreator import from_dict_to_tensor
from .datasetcreator import cifar10_dataset
import torch
import numpy as np
def dataset_init(train_data=True, sample_ratio=1, transformer=None):
    if not train_data:
        data_dict = unpickle('./data/cifar-10-batches-py/test_batch')
        feature, labels = from_dict_to_tensor(data_dict)
        return cifar10_dataset(feature, labels)
    if train_data:
        fea1, lb1 = from_dict_to_tensor(unpickle('./data/cifar-10-batches-py/data_batch_1'))
        fea2, lb2 = from_dict_to_tensor(unpickle('./data/cifar-10-batches-py/data_batch_2'))
        fea3, lb3 = from_dict_to_tensor(unpickle('./data/cifar-10-batches-py/data_batch_3'))
        fea4, lb4 = from_dict_to_tensor(unpickle('./data/cifar-10-batches-py/data_batch_4'))
        fea5, lb5 = from_dict_to_tensor(unpickle('./data/cifar-10-batches-py/data_batch_5'))
        feature = torch.cat([fea1, fea2, fea3, fea4, fea5])
        labels = torch.cat([lb1, lb2, lb3, lb4, lb5])
        sample_num = len(labels)
        sampler = np.random.choice(sample_num, int(sample_num*sample_ratio), replace=False)
        feature = feature[sampler]
        labels = labels[sampler]
        if transformer:
            feature = torch.cat([feature, transformer(feature)])
            labels = torch.cat([labels, labels])
        return cifar10_dataset(feature, labels)
