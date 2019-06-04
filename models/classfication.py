import torch
from torch.nn import functional as F
from torch import nn

# 定义LeNet5
class LeNet5(nn.Module):
    # 构造函数，定义网络的结构
    def __init__(self):
        super().__init__()
        # 定义卷积层，1个输入通道，6个输出通道，5*5的卷积filter，外层补上了两圈0
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        # 定义卷积层，6个输入通道，16个输出通道，5*5的卷积filter
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 最后是三个全连接层
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 最后分为10类

    def forward(self, x):
        # 前向传播函数
        x = F.max_pool2d(F.relu(self.conv1(x)),(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2, 2))
        x = x.view(-1, self.num_flat_features(x))
        # 第一个全连接
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # x.size()返回值为(256, 16, 5, 5)，size的值为(16, 5, 5)，256是batch_size
        size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

