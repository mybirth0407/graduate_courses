# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2021-11-25 02:20:11
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2021-11-29 15:12:05

import torch
import random
import numpy as np
from tqdm import tqdm

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def main():
    K = 600
    N = 3000
    p = 18/37
    q = 0.55

    S = 0
    for repeat in tqdm(range(N)):
        seq = torch.bernoulli(torch.Tensor([q] * K))
        s = torch.sum(seq).numpy()
        f = ((p**s) * ((1-p)**(K-s)))
        g = ((q**s) * ((1-q)**(K-s)))
        S += (f/g) * pi(seq)
    print(S / N)

def pi(X):
    s = 0
    win = False
    for x in X:
        s += 1 if x == 1 else -1
        if s >= 100:
            return True
    return False

if __name__ == '__main__':
    main()
