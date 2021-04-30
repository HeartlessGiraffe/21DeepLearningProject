import torch
from torch import nn
import torch.nn.functional as F

# 定义残差块，每个残差块中有两层卷积层，根据ResNet论文中图片，共需要4个残差块
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, actfun=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        # 残差块中连续的两层卷积层，此处设置偏置项为0
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            actfun,
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        # 判断是否需要进行卷积操作使shortcut连接的x与卷积层输出的维度一致
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        else:
            self.shortcut = nn.Sequential()

    # 前向求算
    def forward(self, x):
        out = self.left(x)
        # 将两个卷积层处理后的输出与原x相加
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10, actfun='relu'):
        super(ResNet, self).__init__()
        self.inchannel = 64
        # 第一层卷积层，对原始图像进行卷积操作，原文献中采用了7*7的卷积核，但由于CIFAR-10数据集中的图像均为32*32*3
        # 直接使用7*7的卷积核会损失较多图像信息，此处采用3*3的卷积核
        # 输入32*32*3，输出32*32*64
        if actfun == 'relu':
            act = nn.ReLU()
        elif actfun == 'sigmoid':
            act = nn.Sigmoid()
        elif actfun == 'tanh':
            act = nn.Tanh()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            act,
        )

        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1, actfun=act)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2, actfun=act)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2, actfun=act)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2, actfun=act)
        self.avg_pool2d = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

    # 由于ResNet-18网络的残差块中需要实现下采样，故需要在添加残差块时调整卷积核的步长
    def make_layer(self, block, outchannel, num_blocks, stride, actfun):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, outchannel, stride, actfun))
            self.inchannel = outchannel
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 原文中使用平均池化后与全连接层进行连接
        # out = self.max_pool2d(out)
        out = self.avg_pool2d(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
