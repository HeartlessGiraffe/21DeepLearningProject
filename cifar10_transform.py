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
    "-t", "--transform", 
    help="type of transform: \n 0-notransformation; 1-HorizontalFlip; 2-Erasing; 3-notransformation+halftrainset",
    type=int,
    choices=[0, 1, 2, 3],
    default=0)

args = parser.parse_args()

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

# 0-notransformation; 1-HorizontalFlip; 2-Erasing; 3-notransformation+halftrainset
transform_type = args.transform

# 两种变形函数
transform_pic_1 = torchvision.transforms.RandomHorizontalFlip(p=1)
transform_pic_2 = torchvision.transforms.RandomErasing(
    p=1, 
    scale=(0.05, 0.33), 
    ratio=(0.3, 3.3))


if __name__ == '__main__':
    test_set = dataset_init(train_data=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    if transform_type == 0:
        train_set = dataset_init()
    elif transform_type == 1:
        train_set = dataset_init(sample_ratio=0.5, transformer=transform_pic_1)
    elif transform_type == 2:
        train_set = dataset_init(sample_ratio=0.5, transformer=transform_pic_2)
    elif transform_type == 3:
        train_set = dataset_init(sample_ratio=0.5)
        num_epoch = num_epoch * 2
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    start_time = datetime.now()
    net = ResNet(ResidualBlock).to(device)
    writer_train_loss_lr2 = SummaryWriter('./runs/train_{}/train_loss_lr2'.format(transform_type))
    writer_test_acc_lr2 = SummaryWriter('./runs/test_{}/test_acc_lr2'.format(transform_type))
    loss_function = nn.CrossEntropyLoss()
    optimizer2 = torch.optim.SGD(net.parameters(), lr=lr)
    print('transform_type:', transform_type)
    for epoch in range(num_epoch):
        if epoch > num_epoch * 2/3 :
            lr = lr * 0.09
        elif epoch > num_epoch * 1/3:
            lr = lr * 0.3
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr
        if continue_training and epoch + 1 < continue_from_epoch:
            print('Epoch {} jumped.'.format(epoch+1))
            continue
        if continue_training:
            net.load_state_dict(
                torch.load('./check_points_{}/model_epoch_{}.pth'.format(transform_type, continue_from_epoch-1)))
        print('\nEpoch: %d' % (epoch + 1), ',lr: ', lr)
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_loader, 0):
            length = len(train_loader)
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.long()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer2.zero_grad()

            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer2.step()

            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).sum()
            if i % 100 == 99:
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (i + 1), sum_loss / (i + 1), 100. * correct / total))
            # train_loss.append(sum_loss / (i + 1))
            writer_train_loss_lr2.add_scalar('training loss',
                                             sum_loss / (i + 1),
                                             epoch * len(train_loader) + i)
        # train_accu.append(correct / total)

        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                net.eval()
                images, labels = data
                images = images.float()
                labels = labels.long()
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        writer_test_acc_lr2.add_scalar('test accuracy',
                                       correct / total,
                                       epoch + 1)
        # accuracy.append(correct / total)

        print('Test\'s accuracy is: %.3f%%' % (100 * correct / total))
        torch.save(net.state_dict(), './check_points/model_{}_epoch_{}.pth'.format(transform_type, epoch+1))
        print('Model trained to epoch {} has been saved.'.format(epoch+1))

    end_time = datetime.now()
    run_time = end_time - start_time

    print('Train has finished, total epoch is %d' % num_epoch)
    print(run_time)
    #
    # plt.figure(figsize=(10, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, len(train_loss) + 1), train_loss)
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # matplotlib.rcParams['axes.unicode_minus'] = False
    # plt.xlabel("迭代次数/次")
    # plt.ylabel("训练集损失")
    # plt.title("训练集损失-迭代次数")
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, len(accuracy) + 1), accuracy,
    #          marker='o', mec='r', mfc='w', label='test_accuracy')
    # plt.plot(range(1, len(train_accu) + 1), train_accu,
    #          marker='*', ms=10, label='train_accuracy')
    # plt.legend()
    # plt.xlabel("epoch")
    # plt.ylabel("准确率")
    # plt.title("训练集与测试集准确率-epoch")
    # plt.show()
