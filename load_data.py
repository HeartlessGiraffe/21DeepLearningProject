from feeder import unpickle
from feeder import from_dict_to_tensor
from feeder import cifar10_dataset
from feeder import dataset_init
import torchvision.transforms as transforms
import torch
import torchvision
from utils import imshow
import random
import numpy as np
if __name__ == '__main__':
    seed = 3417
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    transform_pic_1 = torchvision.transforms.RandomHorizontalFlip(p=1)
    transform_pic_2 = torchvision.transforms.RandomErasing(
        p=1, 
        scale=(0.05, 0.33), 
        ratio=(0.3, 3.3))
    classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # img = torchvision.utils.make_grid(data_feature[2])
    # imshow([img])
    batch_size = 4
    # print(trainset[0])
    trainset = dataset_init(train_data=True, sample_ratio=1, transformer=None)
    print(trainset.count)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=4)
    print('loaded')
    # for data, i in enumerate(trainloader, 0):
    #     print(i)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # print labels
    print(' '.join('%4s' % classes[labels[j]] for j in range(batch_size)))
    # show images
    img = torchvision.utils.make_grid(images)
    img_transformed = torchvision.utils.make_grid(transform_pic_2(images))
    imshow([[img, 'Original'], [img_transformed, 'Transformed']])