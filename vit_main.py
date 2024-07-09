import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset

import vit_model


# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
batch_size = 4

transform_train = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0.25),
     transforms.RandomVerticalFlip(p=0.25),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=mean, std=std)]
)
transform_test = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=mean, std=std)]
)

trainSet = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainLoader = DataLoader(dataset=trainSet, shuffle=True, batch_size=batch_size)

testSet = torchvision.datasets.CIFAR10(root=r'./data', train=False, download=True, transform=transform_test)
# testLoader = DataLoader(dataset=testSet, shuffle=False, batch_size=batch_size)

# 全部数据集太大了，选一部分
train_subset_size = 500
test_subset_size = 50

train_index = np.random.choice(len(trainSet), train_subset_size, replace=False)
test_index = np.random.choice(len(testSet), test_subset_size, replace=False)

trainSubSet = Subset(trainSet, train_index)
testSubSet = Subset(testSet, test_index)

trainLoader = DataLoader(trainSubSet, shuffle=True, batch_size=batch_size)
testLoader = DataLoader(testSubSet, shuffle=False, batch_size=batch_size)

# 查看数据集
def imageshow(img_tensor):
    img = torch.zeros_like(img_tensor)
    for i in range(3):
        img[i] = img_tensor[i] * std[i] + mean[i]
    # img = img_tensor * 0.5 + 0.5

    img_np = img.numpy()
    plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.show()


def create_vit_model():
    return vit_model.vit_base_patch16_224(num_classes=10)


model = create_vit_model()
# print(model)

# 加载部分权重
checkpoint = torch.load(r'vit_base_patch16_224.pth')
# 具体网络层的名字可以点进调试里面看，隐藏了就展开
checkpoint.pop('head.weight', None)  # 去掉Head的权重，要换成num_classes=10
checkpoint.pop('head.bias', None)

model_state_dict = model.state_dict()
model_state_dict.update(checkpoint)
model.load_state_dict(model_state_dict)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), momentum=0.09, lr=0.001)


def mytrain(epochs=10, print_freq=200):
    for epoch in range(epochs):
        loss_sum = 0.0
        for i, data in enumerate(trainLoader, 0):
            images, labels = data
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss
            if i % print_freq == print_freq-1:
                print(f'[{epoch+1}, {i+1}] loss: {loss_sum / print_freq:.3f}')
                loss_sum = 0.0

    print("Training Finished !")


def mytest():  # 注意不能叫test...
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = model(images)
            predicted = torch.argmax(outputs, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()

    print(f'准确率：{100. * correct / total:.3f}')


mytrain(epochs=10, print_freq=50)
mytest()


# 展示数据集
# data_iter = iter(trainLoader)
# images, labels = next(data_iter)
# imageshow(torchvision.utils.make_grid(images))
