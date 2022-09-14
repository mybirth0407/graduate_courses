# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2021-09-23 03:10:30
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2021-09-29 03:34:19

import numpy as np
import time
import matplotlib.pyplot as plt
import random
import copy
from torchsummary import summary

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


class MLP(nn.Module):
    def __init__(self,
        name='3-layer-mlp',
        xdim=1,
        h1dim=64,
        h2dim=64,
        ydim=1
    ):
        super(MLP, self).__init__()
        self.name = 'mlp'
        self.xdim = xdim
        self.h1dim = h1dim
        self.h2dim = h2dim
        self.ydim = ydim

        self.l1 = nn.Linear(xdim, h1dim)
        self.l2 = nn.Linear(h1dim, h2dim)
        self.l3 = nn.Linear(h2dim, ydim)

        self.init_params()

    def init_params(self):
        self.l1.weight.data = torch.normal(
            mean=0, std=1, size=self.l1.weight.shape)
        self.l1.bias.data = torch.full(self.l1.bias.shape, 0.03)
        self.l2.weight.data = torch.normal(
            mean=0, std=1, size=self.l2.weight.shape)
        self.l2.bias.data = torch.full(self.l2.bias.shape, 0.03)
        self.l3.weight.data = torch.normal(
            mean=0, std=1, size=self.l3.weight.shape)
        self.l3.bias.data = torch.full(self.l3.bias.shape, 0.03)

    def forward(self, x):
        net = x
        net = self.l1(net)
        net = torch.sigmoid(net)
        net = self.l2(net)
        net = torch.sigmoid(net)
        net = self.l3(net)
        return net


def main():
    set_seed(0)

    alpha = 0.1
    K = 1000
    B = 128
    N = 512

    X_train = np.random.normal(loc=0.0, scale=1.0, size=N)
    y_train = f_true(X_train) + np.random.normal(0, 0.5, size=X_train.shape)
    X_val = np.random.normal(loc=0.0, scale=1.0, size=N//5)
    y_val = f_true(X_val)

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.float)
    y_val = torch.tensor(y_val, dtype=torch.float)

    train_dataloader = DataLoader(
        TensorDataset(X_train.unsqueeze(1), y_train.unsqueeze(1)),
        batch_size=B, pin_memory=True, shuffle=True
    )

    valid_dataloader = DataLoader(
        TensorDataset(X_val.unsqueeze(1), y_val.unsqueeze(1)),
        batch_size=B, pin_memory=True, shuffle=True
    )
    dataloaders = {
        'train': train_dataloader,
        'valid': valid_dataloader
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ft = MLP().to(device)
    criterion_ft = nn.MSELoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=alpha)
    model_ft, valid_loss_history = train_model(
        model_ft, dataloaders, criterion_ft, optimizer_ft, K, device
    )

    y_preds = []
    with torch.no_grad():
        for inputs, gts in dataloaders['valid']:
            batch_preds = model_ft(inputs.to(device))
            batch_preds = batch_preds.cpu().numpy().tolist()
            y_preds.extend(batch_preds)
            plt.plot(inputs, gts, 'bo', label='ground truth')
            plt.plot(inputs, y_preds, 'rx', label='prediction')
    plt.legend()
    plt.show()

    summary(model_ft, (1, ))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def f_true(x):
    return (x-2) * np.cos(x*4)


def train_model(model, dataloaders, criterion, optimizer, n_epochs,
                device='cpu'):

    valid_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 999999999

    for epoch in range(n_epochs):
        epoch_since = time.time()
        print('Epoch {}/{}'.format(epoch+1, n_epochs))
        print('-' * 10)

        for phase in ['train']:
            if phase == 'train':
                model.train()

            running_loss = 0.0

            for inputs, gts in dataloaders[phase]:
                inputs = inputs.to(device)
                gts = gts.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs.double(), gts.double())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()

            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size

            epoch_time_elapsed = time.time() - epoch_since
            print(
                f'{phase} ({dataset_size}) '
                + f'Loss: {epoch_loss:.4f} '
                + f'Elapsed time: {epoch_time_elapsed // 60:.0f}m '
                + f'{epoch_time_elapsed % 60:.0f}s'
            )

        print()

    return model, valid_loss_history

if __name__ == '__main__':
    main()
