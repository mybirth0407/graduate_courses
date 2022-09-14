# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2021-09-14 04:15:09
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2021-10-30 01:18:17


import numpy as np
import matplotlib.pyplot as plt


def main():
    # data given from hw2
    N, p = 30, 20
    np.random.seed(0)
    X = np.random.randn(N, p)
    Y = 2 * np.random.randint(2, size=N) - 1

    # hyperparameters
    K = 20000
    alpha = 0.007

    # optimization target parameter
    theta = np.random.normal(loc=0.0, scale=4.0, size=p)

    for k in range(K):
        # generates a random sample from X with uniform distribution
        i_k = np.random.choice(np.arange(N), replace=True)
        loss = loss_th(X[i_k], Y[i_k], theta)
        theta = stochastic_gradient_descent(
            X[i_k], Y[i_k],
            lr=alpha,
            theta=theta
        )
        if (k+1) % (K/5) == 0:
            plt.plot(
                k, loss,
                'o',
                label=f'loss = {loss:.4f} after {k+1} iterations'
            )

    plt.legend()
    # plt.show()
    plt.savefig('logisticRegressionSGD.png')


def loss_th(x, y, theta):
    return np.log(1 + np.exp(-y * x@theta))


def diff_loss_th(x, y, theta):
    return (-y * x) / (1 + np.exp(-y * x@theta))


def stochastic_gradient_descent(x, y, lr, theta):
    # l_theta (X_ik, y_ik)
    diff_loss = diff_loss_th(x, y, theta)
    new_theta = theta - lr * diff_loss
    return new_theta


if __name__ == '__main__':
    main()
