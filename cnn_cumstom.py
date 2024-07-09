import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
batch_size = 4

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0.25),
     transforms.RandomVerticalFlip(p=0.25),
     transforms.ToTensor(),
     transforms.Normalize(mean=mean, std=std)]
)

trainSet = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainLoader = DataLoader(dataset=trainSet, shuffle=True, batch_size=batch_size)

testSet = torchvision.datasets.CIFAR10(root=r'./data', train=False, download=True, transform=transform)
testLoader = DataLoader(dataset=testSet, shuffle=False, batch_size=batch_size)


# 查看数据集
def imageshow(img_tensor):
    img = torch.zeros_like(img_tensor)
    for i in range(3):
        img[i] = img_tensor[i] * std[i] + mean[i]
    # img = img_tensor * 0.5 + 0.5

    img_np = img.numpy()
    plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.show()


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.pool(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.pool(x)
        x = x.reshape(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.softmax(x, dim=1)  # nn.CrossEntropy自带softmax
        return x


model = MyNet()
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
                print(f'[{epoch+1}, {i+1} ] loss: {loss_sum / print_freq:.3f}')
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


mytrain(epochs=10, print_freq=1000)
mytest()


# 展示数据集
# data_iter = iter(trainLoader)
# images, labels = next(data_iter)
# imageshow(torchvision.utils.make_grid(images))


#
# data_iter = iter(trainLoader)
# images, labels = next(data_iter)
# output = model(images)
# print(output.shape)






