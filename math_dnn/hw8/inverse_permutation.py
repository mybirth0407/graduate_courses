# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2021-11-23 21:04:10
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2021-11-28 18:01:53

import random
import numpy as np

random.seed(0)
np.random.seed(0)
n = 10

# For convenience, index range changed 1 ~ n to 0 ~ n-1.
sigma =[p for p in np.random.permutation(n)]
inverse_sigma = [-1] * n
print(f'sigma:\n{sigma}')

for i in range(n):
    key = i
    for _ in range(n):
        key = sigma[key]
        if key == i:
            inverse_sigma[sigma[key]] = i
            break

print(f'inverse sigma:\n{inverse_sigma}')