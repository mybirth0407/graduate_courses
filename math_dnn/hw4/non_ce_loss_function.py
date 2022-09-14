# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2021-10-05 02:08:20
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2021-10-07 05:21:59

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils import data as data_utils
from torchvision import datasets
from torchvision.transforms import transforms
from torchsummary import summary
from typing import NoReturn
import random
import numpy as np
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LogisticRegression(nn.Module):
    def __init__(self,
        input_dim: int = 1*28*28,
        output_dim: int = 1
    ) -> NoReturn:

        super(LogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear1 = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear1(x.float().view(-1, self.input_dim))
        return out


def main():
    set_seed(0)
    B = 100

    # MNIST training data
    train_dataset = datasets.MNIST(
        root='./mnist_data/',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    # MNIST testing data
    test_dataset = datasets.MNIST(
        root='./mnist_data/',
        train=False,
        transform=transforms.ToTensor()
    )

    label_1, label_2 = 4, 9
    # Use data with two labels
    idx = (train_dataset.targets == label_1) + (train_dataset.targets == label_2)
    train_dataset.data = train_dataset.data[idx]
    train_dataset.targets = train_dataset.targets[idx]
    train_dataset.targets[train_dataset.targets == label_1] = -1
    train_dataset.targets[train_dataset.targets == label_2] = 1

    # Use data with two labels
    idx = (test_dataset.targets == label_1) + (test_dataset.targets == label_2)
    test_dataset.data = test_dataset.data[idx]
    test_dataset.targets = test_dataset.targets[idx]
    test_dataset.targets[test_dataset.targets == label_1] = -1
    test_dataset.targets[test_dataset.targets == label_2] = 1

    model = LogisticRegression().to(device)
    loss_function = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    
    train_loader = data_utils.DataLoader(
        dataset = train_dataset,
        batch_size = B,
        shuffle = True,
    )

    start = time.time()
    for epoch in range(10) :
        print('{}th epoch starting.'.format(epoch+1))
        for images, labels in train_loader :
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            optimizer.zero_grad()
            train_loss = loss_function(outputs[:, 0], labels.float())
            train_loss.backward()
            optimizer.step()
    end = time.time()
    print('Time ellapsed in training is: {}'.format(end - start))

    test_loss, correct, total = 0, 0, 0

    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = 100,
        shuffle = False
    )

    for images, labels in test_loader :
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        test_loss += loss_function(outputs[:, 0], labels.float()).item()
        correct += torch.sum(outputs[:, 0] * labels.float() >= 0)
        total += labels.size(0)

    print(
        '[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
        .format(test_loss / total, correct, total, 100. * correct / total)
    )


def non_ce_criterion(z, y):
    return (
        1/2 * (1 - y) * ((1 - torch.sigmoid(-z))**2)
        + 1/2 * (1 + y) * ((torch.sigmoid(-z))**2 + (1-torch.sigmoid(z))**2)
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def logistic_loss(output, target):
    return -torch.nn.functional.logsigmoid(target*output)


if __name__ == '__main__':
    main()