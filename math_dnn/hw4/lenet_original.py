# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2021-09-30 18:33:36
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2021-10-07 04:32:12

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torchsummary import summary
from typing import NoReturn
import random
import numpy as np
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
Step 2: LeNet5
"""
# Modern LeNet uses this layer for C3
class C3_layer_full(nn.Module):
    def __init__(self):
        super(C3_layer_full, self).__init__()
        self.conv_layer = nn.Conv2d(6, 16, kernel_size=5)

    def forward(self, x):
        return self.conv_layer(x)


# Original LeNet uses this layer for C3
class C3_layer(nn.Module):
    def __init__(self,
        kernel_size: int = 5,
    ) -> NoReturn:
        super(C3_layer, self).__init__()
        self.ch_in_3 = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [0, 4, 5],
            [0, 1, 5]
        ] # filter with 3 subset of input channels

        self.ch_in_4 = [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [0, 3, 4, 5],
            [0, 1, 4, 5],
            [0, 1, 2, 5],
            [0, 1, 3, 4],
            [1, 2, 4, 5],
            [0, 2, 3, 5],
        ] # filter with 4 subset of input channels

        # put implementation here
        self.ch_in_6 = [
            [0, 1 ,2, 3, 4, 5]
        ]

        # No efficient code, but more intuitive.
        self.conv_layer = nn.ModuleList([
            nn.Conv2d(3, 1, kernel_size=kernel_size),
            nn.Conv2d(3, 1, kernel_size=kernel_size),
            nn.Conv2d(3, 1, kernel_size=kernel_size),
            nn.Conv2d(3, 1, kernel_size=kernel_size),
            nn.Conv2d(3, 1, kernel_size=kernel_size),
            nn.Conv2d(3, 1, kernel_size=kernel_size),
            nn.Conv2d(4, 1, kernel_size=kernel_size),
            nn.Conv2d(4, 1, kernel_size=kernel_size),
            nn.Conv2d(4, 1, kernel_size=kernel_size),
            nn.Conv2d(4, 1, kernel_size=kernel_size),
            nn.Conv2d(4, 1, kernel_size=kernel_size),
            nn.Conv2d(4, 1, kernel_size=kernel_size),
            nn.Conv2d(4, 1, kernel_size=kernel_size),
            nn.Conv2d(4, 1, kernel_size=kernel_size),
            nn.Conv2d(4, 1, kernel_size=kernel_size),
            nn.Conv2d(6, 1, kernel_size=kernel_size),
        ])

        # # Pytorch efficient code
        # self.conv_layer = nn.ModuleList([
        #     nn.Conv2d(3, 6, kernel_size=kernel_size),
        #     nn.Conv2d(4, 9, kernel_size=kernel_size),
        #     nn.Conv2d(6, 1, kernel_size=kernel_size)
        # ])

    # torch.Size([100, 6, 14, 14] to torch.Size([100, 16, 10, 10])
    def forward(self, x):
        # No efficient code, but more intuitive.
        out = torch.cat([
            self.conv_layer[0](x[:, self.ch_in_3[0], :, :]),
            self.conv_layer[1](x[:, self.ch_in_3[1], :, :]),
            self.conv_layer[2](x[:, self.ch_in_3[2], :, :]),
            self.conv_layer[3](x[:, self.ch_in_3[3], :, :]),
            self.conv_layer[4](x[:, self.ch_in_3[4], :, :]),
            self.conv_layer[5](x[:, self.ch_in_3[5], :, :]),
            self.conv_layer[6](x[:, self.ch_in_4[0], :, :]),
            self.conv_layer[7](x[:, self.ch_in_4[1], :, :]),
            self.conv_layer[8](x[:, self.ch_in_4[2], :, :]),
            self.conv_layer[9](x[:, self.ch_in_4[3], :, :]),
            self.conv_layer[10](x[:, self.ch_in_4[4], :, :]),
            self.conv_layer[11](x[:, self.ch_in_4[5], :, :]),
            self.conv_layer[12](x[:, self.ch_in_4[6], :, :]),
            self.conv_layer[13](x[:, self.ch_in_4[7], :, :]),
            self.conv_layer[14](x[:, self.ch_in_4[8], :, :]),
            self.conv_layer[15](x[:, self.ch_in_6[0], :, :]),
        ], dim=1)

        # # pytorch efficient code
        # out = torch.cat([
        #     self.conv_layer[0](torch.sum(x[:, self.ch_in_3, :, :], dim=1)),
        #     self.conv_layer[1](torch.sum(x[:, self.ch_in_4, :, :], dim=1)),
        #     self.conv_layer[2](torch.sum(x[:, self.ch_in_6, :, :], dim=1))
        # ], dim=1)

        return out

class LeNet(nn.Module) :
    def __init__(self) :
        super(LeNet, self).__init__()

        # padding=2 makes 28x28 image into 32x32
        self.C1_layer = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh()
        ) # 1 * 32 * 32 -> 6 * 28 * 28

        self.P2_layer = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Tanh()
        ) # 6 * 28 * 28 -> 6 * 14 * 14

        # this layer makes 10x10 image into 14x14
        self.C3_layer = nn.Sequential(
            C3_layer_full(),
            # C3_layer(),
            nn.Tanh()
        ) # 6 * 14 * 14 -> 16 * 10 * 10

        self.P4_layer = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Tanh()
        ) # 16 * 10 * 10 -> 16 * 5 * 5

        self.C5_layer = nn.Sequential(
            nn.Linear(5*5*16, 120),
            nn.Tanh()
        ) # 16 * 5 * 5 -> 1 * 120

        self.F6_layer = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh()
        ) # 120 -> 84

        self.F7_layer = nn.Linear(84, 10) # 84 -> 10
        self.tanh = nn.Tanh() # 10 -> 10 elem-wise tanh

    def forward(self, x) :
        out = self.C1_layer(x)
        out = self.P2_layer(out)
        out = self.C3_layer(out)
        out = self.P4_layer(out)

        out = out.view(-1, 5*5*16)
        out = self.C5_layer(out)
        out = self.F6_layer(out)
        out = self.F7_layer(out)

        return out


def main():
    set_seed(0)
    
    """
    Step 1:
    """
    # MNIST dataset
    train_dataset = datasets.MNIST(
        root='./mnist_data/',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    test_dataset = datasets.MNIST(
        root='./mnist_data/',
        train=False,
        transform=transforms.ToTensor()
    )

    """
    Step 3
    """
    model = LeNet().to(device)
    summary(model, (1, 28, 28))

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # print total number of trainable parameters
    param_ct = sum([p.numel() for p in model.parameters()])
    print(f'Total number of trainable parameters: {param_ct}')

    """
    Step 4
    """
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = 100,
        shuffle = True
    )

    start = time.time()
    for epoch in range(10) :
        print('{}th epoch starting.'.format(epoch+1))
        for images, labels in train_loader :
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            train_loss = loss_function(model(images), labels)
            train_loss.backward()

            optimizer.step()
    end = time.time()
    print('Time ellapsed in training is: {}'.format(end - start))


    """
    Step 5
    """
    test_loss, correct, total = 0, 0, 0

    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = 100,
        shuffle = False
    )

    for images, labels in test_loader :
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        test_loss += loss_function(output, labels).item()

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.size(0)

    print(
        '[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
        .format(test_loss / total, correct, total, 100. * correct / total)
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


if __name__ == '__main__':
    main()