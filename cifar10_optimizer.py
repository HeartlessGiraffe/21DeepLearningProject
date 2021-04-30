import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from model import ResNet
from model import ResidualBlock
from torch.utils.tensorboard import SummaryWriter

'''搭建ResNet18神经网络实现CIFAR-10数据集的图像分类'''
# metrics = {
#     'accuracy: ': accuracy_score,
#     'precision: ': lambda x, y: precision_score(x, y, average='macro'),
#     'recall: ': lambda x, y: recall_score(x, y, average='macro'),
#     'confustion matrix: \n': confusion_matrix
# }

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设定超参数
num_epoch = 10
batch_size = 100

# 数据下载与预处理
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.1 * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    # 用于实验的三种优化器
    optims = ['Adam', 'Adagrad', 'SGD']
    for optim in optims:
        lr0 = 0.1
        actfun = 'sigmoid'
        # 计算实验耗时
        start_time = datetime.now()
        net = ResNet(ResidualBlock, actfun=actfun).to(device)
        print(net)
        # 创建tensorboard文件夹
        writer_train_loss = SummaryWriter('./runs/train/' + actfun)
        writer_test_acc = SummaryWriter('./runs/test/' + actfun)

        # 损失函数使用交叉熵损失函数
        loss_function = nn.CrossEntropyLoss()

        # train_loss = []
        # train_accu = []
        # accuracy = []
        if optim == 'Adam':
            optimizer0 = torch.optim.Adam(net.parameters(), lr=lr0, betas=(0.9, 0.99))
        elif optim == 'Adagrad':
            optimizer0 = torch.optim.Adagrad(net.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
        elif optim == 'SGD':
            optimizer0 = torch.optim.SGD(net.parameters(), lr=lr0)

        for epoch in range(num_epoch):
            print('\nEpoch: %d' % (epoch + 1))
            if epoch > 0 and epoch % 3 == 0:
                lr0 *= 0.1
                print('learning_rate=', lr0)
            
            adjust_learning_rate(optimizer0, epoch)
            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            # 每个i对应一个iteration
            for i, data in enumerate(train_loader, 0):
                length = len(train_loader)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # 将模型的参数梯度初始化为0
                optimizer0.zero_grad()

                outputs = net(inputs)
                # 下三行代码用于输出网络结构，但是效果不是很好，隐去
                # with SummaryWriter(comment='ResNet-18') as w:
                #     w.add_graph(net, (inputs,))
                # print("ResNet:", outputs.shape)
                # 求算损失
                loss = loss_function(outputs, labels)
                # 反向传播
                loss.backward()
                # 更新参数
                optimizer0.step()

                sum_loss += loss.item()
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).sum()
                if i % 100 == 99:
                #     print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                #           % (epoch + 1, (i + 1), sum_loss / (i + 1), 100. * correct / total))
                # # train_loss.append(sum_loss / (i + 1))
                # writer_train_loss_lr0.add_scalar('training loss',
                #                                  sum_loss / (i + 1),
                #                                  epoch * len(train_loader) + i)
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (i + 1), loss.item(), 100. * correct / total))
                # 也可以用如下的平均损失替代
                # train_loss.append(sum_loss / (i + 1))
                # 每个iteration向tensorboard日志中写入一次训练loss
                writer_train_loss.add_scalar('training loss',
                                             loss.item(),
                                             epoch * len(train_loader) + i)
            # train_accu.append(correct / total)
            # 进行测试时不需要求算梯度，提高运算速度
            with torch.no_grad():
                correct = 0
                total = 0
                for data in test_loader:
                    net.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    # 取得分最高的那个类 (outputs.data的索引号)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
            # 每个epoch向tensorboard中写入一次测试集准确率
            writer_test_acc.add_scalar('test accuracy',
                                        correct / total,
                                        epoch + 1)
            # accuracy.append(correct / total)

            print('Test\'s accuracy is: %.3f%%' % (100 * correct / total))
        # 计算结束时间
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
