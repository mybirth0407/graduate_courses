# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2021-10-08 15:56:42
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2021-10-18 16:18:31

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils import data as data_utils
from torchvision import datasets
from torchvision.transforms import transforms
from torchsummary import summary
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Make sure to use only 10% of the available MNIST data.
# Otherwise, experiment will take quite long (around 90 minutes).

# (Modified version of AlexNet)
class AlexNet(nn.Module):
    def __init__(self, num_class=10):
        super(AlexNet, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fc_layer1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6400, 800),
            nn.ReLU(inplace=True),
            nn.Linear(800, 10)
        )

    def forward(self, x):
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = torch.flatten(output, 1)
        output = self.fc_layer1(output)
        return output


def main():
    set_seed(0)

    # MNIST training data
    train_dataset = datasets.MNIST(
        root='./mnist_data/',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    idx = torch.randperm(train_dataset.data.shape[0])
    train_dataset.data = (
        train_dataset.data[idx].view(train_dataset.data.size())
    )
    train_dataset.data = (
        train_dataset.data[: train_dataset.data.shape[0]//10, :]
    )
    train_dataset.targets = (
        torch.randint(low=0, high=10, size=train_dataset.targets.shape)
    )

    learning_rate = 0.1
    batch_size = 64
    epochs = 150

    model = AlexNet().to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True
    )

    train_losses = []
    train_acc = []
    tick = time.time()

    for epoch in tqdm(range(epochs), total=epochs):
        # print(f'Epoch {epoch + 1} / {epochs}')
        epoch_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            optimizer.zero_grad()
            loss = loss_function(output, labels)
            epoch_loss += loss.detach().cpu().numpy()
            loss.backward()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)

            optimizer.step()
            
        epoch_loss /= len(train_loader)
        epoch_acc = correct / total
        # print(f'loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}')
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)
    tock = time.time()

    print(f"Total training time: {tock - tick}")

    t = range(epochs)

    plt.figure(figsize=(15, 10))
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    lns1 = ax1.plot(t, train_acc, color=color, label='Train Accuracy')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    lns2 = ax2.plot(t, train_losses, color=color, label='Train Loss')

    plt.tight_layout(pad=40)  # otherwise the right y-label is slightly clipped

    lns = lns1 + lns2
    plt.legend(
        lns,
        ['Train Accuracy', 'Train Loss'],
        loc='center left',
        bbox_to_anchor=(0.78, 1.08)
    )
    plt.title('Training with Randomized Label')
    plt.savefig('label_memorization.png', dpi=100)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    main()