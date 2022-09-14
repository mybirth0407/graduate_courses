# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2021-11-30 01:00:55
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2021-11-30 01:08:12

import torch
import random
import numpy as np
import matplotlib.pyplot as plt


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# log derivative
tau1 = torch.tensor([0.])
mu1 = torch.tensor([0.])
lr = 1e-2
B = 16
epochs= 50
mu_history1 = torch.zeros((epochs+1, 1))
tau_history1 = torch.zeros((epochs+1, 1))

for epoch in range(epochs):
    X = torch.normal(torch.zeros((B, 1)), tau1.exp()**2) + mu1
    g = torch.sum((X * torch.sin(X) * (X - mu1) / (tau1.exp()**3)) + mu1 - 1)
    h = torch.sum((X * torch.sin(X) * (3/2 * (X - mu1)**2 / (tau1.exp()**3))) + tau1.exp() - 1)
    mu1 -= lr*g
    tau1 -= lr*h
    # save history
    mu_history1[epoch+1] = mu1
    tau_history1[epoch+1] = tau1.exp()

# reparametrization
tau2 = torch.tensor([0.])
mu2 = torch.tensor([0.])
mu_history2 = torch.zeros((epochs+1, 1))
tau_history2 = torch.zeros((epochs+1, 1))

for epoch in range(epochs):
    Y = torch.normal(0, 1, size=(B,1))
    g = torch.sum((torch.sin(Y * tau2.exp() + mu2) + (Y * tau2.exp() + mu2) * torch.cos(Y * tau2.exp() + mu2)) + mu2 - 1)
    h = torch.sum((Y * tau2.exp()) * (torch.sin(Y * tau2.exp() + mu2) + (Y * tau2.exp() + mu2) * torch.cos(Y * tau2.exp() + mu2)) + tau2.exp() - 1)
    mu2 -= lr*g
    tau2 -= lr*h
    # save history
    mu_history2[epoch+1] = mu2
    tau_history2[epoch+1] = tau2.exp()

x1 = mu_history1.numpy()
y1 = tau_history1.numpy()

x2 = mu_history2.numpy()
y2 = tau_history2.numpy()

plt.subplot(1, 2, 1)
plt.scatter(0,0, s=100, c='green')
plt.scatter(x1[-1], y1[-1], s=100, c='red')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(x1, y1, linestyle='solid',color='blue')
plt.title('log derivative trick')

plt.subplot(1, 2, 2)
plt.scatter(0,0, s=100, c='green')
plt.scatter(x2[-1], y2[-1],  s=100, c='red')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(x2, y2, linestyle='solid',color='blue')
plt.title('reparametrization trick')

plt.tight_layout()
# plt.show()
plt.savefig('derivative_trick.png')