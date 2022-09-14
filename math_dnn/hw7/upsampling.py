# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2021-11-15 23:25:56
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2021-11-16 12:47:12

import torch
from torch import nn

torch.manual_seed(42)
b, cin, cout = 1, 1, 1
m, n, r = 3, 3, 2

input = torch.randn(b, cin, m, n)
print(input)
layer = nn.Upsample(scale_factor=r, mode='nearest')
print(layer(input))

upsample = nn.ConvTranspose2d(cin, cout, kernel_size=r, stride=r, bias=False)
upsample.weight.data = torch.tensor([[[[1.] * r] * r] * cout] * b)
print(upsample(input))