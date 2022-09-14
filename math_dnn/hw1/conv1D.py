# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2021-09-01 23:07:54
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2021-10-24 01:07:20

import numpy as np


class Convolution1d :
    def __init__(self, filt) :
        self.__filt = filt
        self.__r = filt.size
        self.T = TransposedConvolution1d(self.__filt)

    def __matmul__(self, vector) :
        r, n = self.__r, vector.size

        """ solution ver 1: non-pythonic """
        # ret = []
        # for i in range(n-r+1):
        #     v = 0;
        #     for j in range(r):
        #         v += self.__filt[j] * vector[i + j]
        #     ret.append(v)
        # return np.array(ret)

        """ solution ver2: vectorization but not pythonic """
        # return np.array([
        #     sum(self.__filt[j]*vector[i+j] for j in range(r))
        #         for i in range(n-r+1)
        # ])

        """ solution ver3: pythonic and vectorize """
        return np.array([
            self.__filt@vector[i:i+r] for i in range(n-r+1)
        ])


class TransposedConvolution1d :
    """
    Transpose of 1-dimensional convolution operator used for the 
    transpose-convolution operation A.T@(...)
    """

    def __init__(self, filt) :
        self.__filt = filt
        self.__r = filt.size

    def __matmul__(self, vector) :
        r = self.__r
        n = vector.size + r - 1 # 20

        """ solution ver 1: non-pythonic """
        # ret = []
        # for i in range(n):
        #     v = 0
        #     for j in range(r):
        #         x = 0 if i-r+j+1 < 0 or i-r+j+1 >= n-r+1 else vector[i-r+j+1]
        #         v += self.__filt[r-j-1] * x
        #     ret.append(v)
        # return np.array(ret)

        """ solution ver2: vectorization but not pythonic """
        return np.array([
            sum(0 if i-r+j+1 < 0 or i-r+j+1 >= n-r+1
                else self.__filt[r-j-1]*vector[i-r+j+1] for j in range(r)
            ) for i in range(n)
        ])


def main():
    r, n, lam = 3, 20, 0.1
    np.random.seed(0)
    k = np.random.randn(r)
    b = np.random.randn(n-r+1)

    A = Convolution1d(k)

    x = np.zeros(n)

    # for answer test
    # from scipy.linalg import circulant
    # A = circulant(np.concatenate((np.flip(k),np.zeros(n-r))))[2:,:]

    alpha = 0.01
    for _ in range(100) :
        x = x - alpha*(A.T@(huber_grad(A@x-b))+lam*x)

    # before
    # print(huber_loss(A@x-b)+0.5*np.linalg.norm(x*x)**2)
    # after
    print(huber_loss(A@x-b)+0.5*lam*np.linalg.norm(x)**2)

def huber_loss(x) :
    return np.sum(
        (1/2)*(x**2)*(np.abs(x)<=1) + (np.sign(x)*x-1/2)*(np.abs(x)>1)
    )

def huber_grad(x) :
    return x*(np.abs(x)<=1) + np.sign(x)*(np.abs(x)>1)


if __name__ == '__main__':
    main()
