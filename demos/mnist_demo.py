#!usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: shenxudong
@file: mnist_demo.py 
@version: 0.0.1
@time: 2019/06/04 
@email: shenxvdong1@gmail.com
@function： 
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.classfication import LeNet5
import torch.nn as nn
from torch.autograd import Variable

# 读取数据
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])]
)

train_data = datasets.MNIST(
    root="../datasets/mnist",
    train=True,
    transform=data_tf,
    download=False
)

test_data = datasets.MNIST(
    root="../datasets/mnist/",
    train=False,
    transform=data_tf,
    download=False
)

train_dataloader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, )
test_dataloader = DataLoader(dataset=test_data, batch_size=16, shuffle=False, )

# 训练模型
# #  step1. 初始化模型，创建优化器和损失函数
model = LeNet5()

loss_func = nn.CrossEntropyLoss()
# opti = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)
opti = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.01,weight_decay=0.0)

# 检查GPU是否可用
if torch.cuda.is_available():
    model.cuda()


# # step2. 设置函数保存和评估模型
def save_model(epoch):
    torch.save(model.state_dict(), "lenet5_model_{}.model".format(epoch))
    print("Checkpoint Saved")


def mnist_test():
    model.eval()
    test_acc = 0
    for i, (images, labels) in enumerate(test_dataloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_acc += torch.sum(prediction == labels.data)
    return test_acc


# # 训练函数
def train(num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("---------------Start Train-------------------")
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            # 清楚所有累积梯度
            opti.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()

            # 根据计算的梯度
            opti.step()

            train_loss += loss.cpu().data * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data)

        test_acc = mnist_test()
        if test_acc > best_acc:
            save_model(epoch)
            best_acc = test_acc
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {}, Test Accuarcy: {}".format(epoch,
                                                                                       train_acc.item()/len(train_data), train_loss / len(train_data), test_acc.item() / len(test_data)))


if __name__ == '__main__':
    train(50)
