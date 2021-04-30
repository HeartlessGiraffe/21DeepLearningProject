import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from model import ResNet
from model import ResidualBlock
import os
import random
import numpy as np
from feeder import unpickle
from feeder import from_dict_to_tensor
from feeder import cifar10_dataset
from feeder import dataset_init
from torch.utils.tensorboard import SummaryWriter

'''搭建ResNet18神经网络实现CIFAR-10数据集的图像分类'''
# 读取参数
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--activiation", 
    help="type of activiation: \n 0, 1, 2 - sigmoid, tanh, relu \n default relu",
    type=int,
    choices=[0, 1, 2],
    default=2)

args = parser.parse_args()
actfuns = ['sigmoid', 'tanh', 'relu']
actfun = actfuns[args.activiation]

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设定超参数
num_epoch = 10
batch_size = 100
lr = 0.1
continue_training = False
continue_from_epoch = 0
seed = 3417
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


if __name__ == '__main__':
    path = './trained_model/model.pth'
    test_set = dataset_init(train_data=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    net = ResNet(ResidualBlock, actfun=actfun).to(device)
    net.load_state_dict(torch.load(path, map_location=torch.device(device))['state_dict'])
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        i = 0
        for data in test_loader:
            i += 1
            images, labels = data
            images = images.float()
            labels = labels.long()
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            print('batch:', '{}/{}'.format(i, len(test_loader)))
    
    print('Test\'s accuracy is: %.3f%%' % (100 * correct / total))

