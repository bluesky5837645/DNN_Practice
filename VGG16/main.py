import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import random


# TODO: Creat the UI interface by pyqt
# TODO: Add Validation in train process(https://github.com/rasbt/stat453-deep-learning-ss21/tree/main/L14)

# ======================================================
# imshow function
def imshow(img_list, label_list):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    fig = plt.figure()
    for i in range(9):
        fig.add_subplot(3, 3, i+1)
        ax = plt.imshow(np.transpose(img_list[i], (1, 2, 0)))
        plt.title(classes[label_list[i]], fontsize=9)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    plt.show()

# ======================================================
# hyperparameter
batch_size = 256
learning_rate = 1e-2
num_epoches = 20
# ======================================================
# train_dataset
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((70, 70)),
    torchvision.transforms.RandomCrop((64, 64)),
    torchvision.transforms.ToTensor()])

train_dataset = datasets.CIFAR10('./data', train=True, transform=train_transforms, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# ======================================================
# test_dataset
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((70, 70)),
    torchvision.transforms.CenterCrop((64, 64)),
    torchvision.transforms.ToTensor()])
test_dataset = datasets.CIFAR10('./data', train=False, transform=test_transforms, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# ======================================================
# VGG16 model


class VGG16(nn.Module):
    def __init__(self, num_classes = 10):
        super(VGG16, self).__init__()
        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=64,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64,
                            out_channels=64,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2),
                               stride=(2, 2))
        )

        self.block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,
                            out_channels=128,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128,
                            out_channels=128,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2),
                               stride=(2, 2))
        )

        self.block_3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128,
                            out_channels=256,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=256,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=256,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2),
                               stride=(2, 2))
        )

        self.block_4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256,
                            out_channels=512,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512,
                            out_channels=512,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512,
                            out_channels=512,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2),
                               stride=(2, 2))
        )

        self.block_5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512,
                            out_channels=512,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512,
                            out_channels=512,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512,
                            out_channels=512,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2),
                               stride=(2, 2))
        )

        height, width = 3, 3
        self.avgpool = torch.nn.AdaptiveAvgPool2d((height, width))

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * height * width, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, num_classes),
        )
        # ======================================================
        # initial
        for m in self.modules():
            if isinstance(m, torch.torch.nn.Conv2d) or isinstance(m, torch.torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()

    # ======================================================
    # model(image) will call this function
    def forward(self, x):
            x = self.block_1(x)
            x = self.block_2(x)
            x = self.block_3(x)
            x = self.block_4(x)
            x = self.block_5(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            logits = self.classifier(x)

            return logits


if __name__ == '__main__':
    # ======================================================
    # show some train data
    show_img_idx = [random.randint(0, len(train_dataset) - 1) for _ in range(9)]
    imag_list = [train_dataset[indx][0].numpy() for indx in show_img_idx]
    label_list = [train_dataset[indx][1] for indx in show_img_idx]
    imshow(imag_list, label_list)
    # ======================================================
    # creat model
    model = VGG16()

    # ======================================================
    # detect the device
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    # ======================================================
    # optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.01)

    # ======================================================
    # train mode
    model.train()

    # ======================================================
    # record
    mimi_batch_loss = []
    epoches_loss = []
    train_acc = []
    test_acc = []

    # ======================================================
    # train
    for epoch in range(num_epoches):
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)

            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()


            optimizer.zero_grad()
            # ======================================================
            # forward
            y_pred = model(images)

            # ======================================================
            # backward
            loss = criterion(y_pred, labels)

            loss.backward()

            optimizer.step()

            # ======================================================
            # print Iter info
            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (epoch + 1, num_epoches, i + 1, 196, loss.data))
            mimi_batch_loss.append(loss.data.cpu().numpy())

            _, predicted = torch.max(y_pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        # ======================================================
        # record & print Epoch info
        print('Accuracy of the model on the train images: %d %%' % (100 * correct / total))
        train_acc.append((100 * correct / total).cpu().numpy())
        epoches_loss.append(loss.data.cpu().numpy())


        # ======================================================
        # test mode()
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = Variable(images)
                labels = Variable(labels)
                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                y_pred = model(images)
                _, predicted = torch.max(y_pred.data, 1)
                total += labels.size(0)
                temp = (predicted == labels.data).sum()
                correct += temp
        # ======================================================
        # print test info
        print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))
        test_acc.append((100 * correct / total).cpu().numpy())
    # ======================================================
    # save model
    torch.save(model.state_dict(), 'save.pt')

    # ======================================================
    # plot loss and acc
    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    plt.plot(np.linspace(1, num_epoches,num_epoches), np.asarray(train_acc))
    plt.plot(np.linspace(1, num_epoches,num_epoches), np.asarray(test_acc))
    fig.add_subplot(2, 1, 2)
    plt.plot(np.linspace(1, num_epoches,num_epoches), np.asarray(epoches_loss))
    plt.show()



















