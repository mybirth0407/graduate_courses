# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2021-10-08 02:39:22
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2021-10-10 17:15:46

import torch
from torch import nn


def main():
    torch.manual_seed(0)
    L = 6
    X_data = torch.rand(4, 1)
    Y_data = torch.rand(1, 1)

    A_list,b_list = [],[]
    for _ in range(L-1):
        A_list.append(torch.rand(4, 4))
        b_list.append(torch.rand(4, 1))
    A_list.append(torch.rand(1, 4))
    b_list.append(torch.rand(1, 1))


    # Option 1: directly use PyTorch's autograd feature
    for A in A_list:
        A.requires_grad = True
    for b in b_list:
        b.requires_grad = True
        
    y = X_data
    for ell in range(L):
        S = sigma if ell<L-1 else lambda x: x
        y = S(A_list[ell]@y+b_list[ell])
        
    # backward pass in pytorch
    loss=torch.square(y-Y_data)/2
    loss.backward()

    print('Option 1')
    print(A_list[0].grad)
    print(b_list[0].grad)


    # Option 2: construct a NN model and use backprop
    class MLP(nn.Module) :
        def __init__(self) :
            super().__init__()
            self.linear = []
            for ell in range(L-1):
                self.linear.append(nn.Linear(4, 4, bias=True))
            self.linear.append(nn.Linear(4, 1, bias=True))
            for ell in range(L):
                self.linear[ell].weight.data = A_list[ell]
                self.linear[ell].bias.data = b_list[ell].squeeze()

        def forward(self, x) :
            x = x.squeeze()
            for ell in range(L-1):
                x = sigma(self.linear[ell](x))
            x = self.linear[-1](x)
            return x

    model = MLP()
                
    loss = torch.square(model(X_data)-Y_data)/2
    loss.backward()

    print()
    print('Option 2')
    print(model.linear[0].weight.grad)
    print(model.linear[0].bias.grad)


    # Option 3: implement backprop yourself
    y_list = [X_data]
    y = X_data
    for ell in range(L):
        S = sigma if ell<L-1 else lambda x: x
        y = S(A_list[ell]@y + b_list[ell])
        y_list.append(y)


    debug_opt = 0

    dA_list = []
    db_list = []
    dy = y - Y_data
    for ell in reversed(range(L)):
        S = sigma_prime if ell < L-1 else lambda x: torch.ones(x.shape)
        A, b, y = A_list[ell], b_list[ell], y_list[ell]

        db = dy@torch.ones(1, 1).T * S(A@y + b) # dloss/db 1
        if debug_opt:
            debug_print('db', db)
        
        dA = dy@y.T * S(A@y + b) # dloss/dA 2
        if debug_opt:
            debug_print('dA', dA)
        
        dy = A.T@(dy * S(A@y + b))# dloss/dy 3
        if debug_opt:
            debug_print('dy', dy)

        dA_list.insert(0, dA)
        db_list.insert(0, db)

    print()
    print('implement backprop myself')
    print(dA_list[0])
    print(db_list[0])


def sigma(x):
    return torch.sigmoid(x)


def sigma_prime(x):
    return sigma(x)*(1-sigma(x))


def debug_print(name, x):
    print(name)
    print('=' * 10 + 'value' + '=' * 10)
    print(x)
    print('=' * 10 + 'shape' + '=' * 10)
    print(x.shape)

if __name__ == '__main__':
    main()
