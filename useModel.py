from main import VGG16
from matplotlib import pyplot as plt
from torchvision import datasets
import torchvision
import torch
import numpy as np


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# ======================================================
# user define parameter
choose_type = 3


with torch.no_grad():
    # ======================================================
    # creat and load the model
    model = VGG16()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    # ======================================================
    # creat test dataset
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.CenterCrop((64, 64)),
        torchvision.transforms.ToTensor()])
    test_dataset = datasets.CIFAR10('./data', train=False, transform=test_transforms, download=True)

    # ======================================================
    # search the image that we want
    search_ind = 1
    while True:
        imgae_test, label = test_dataset[search_ind]
        if label == choose_type:
            break
        search_ind += 1

    # ======================================================
    # test the model adn plot the result
    imgae_test_input = imgae_test.unsqueeze(0)
    pre = model(imgae_test_input)
    plt.figure()
    plt.subplot(1, 2, 1)
    ax = plt.imshow(np.transpose(imgae_test, (1, 2, 0)))
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title(classes[choose_type], fontsize=21)
    ax = plt.subplot(1, 2, 2)
    plt.bar(x=range(10), height=pre.numpy()[0])
    plt.title('Predict Output', fontsize=21)
    plt.xticks(range(10), list(classes))
    plt.grid()
    plt.show()


