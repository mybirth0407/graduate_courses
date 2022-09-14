# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2021-09-14 04:15:09
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2021-11-05 16:45:37


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
    lambda_ = 0.1
    alpha = 0.0007

    # optimization target parameter
    theta = np.random.normal(loc=0.0, scale=4.0, size=p)

    for k in range(K):
        # generates a random sample from X with uniform distribution
        i_k = np.random.choice(np.arange(N), replace=True)
        loss = loss_th(X[i_k], Y[i_k], theta, lambda_)
        theta = stochastic_gradient_descent(
            X[i_k], Y[i_k],
            lr=alpha,
            theta=theta,
            lambda_=lambda_
        )
        if (k+1) % (K/5) == 0:
            plt.plot(
                k, loss,
                'o',
                label=f'loss = {loss:.4f} after {k+1} iterations'
            )

    plt.legend()
    # plt.show()
    plt.savefig('supportVectorMachineSGD.png')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def loss_th(x, y, theta, lambda_):
    return max(
            0,
            1 - (y * (x@theta))
        ) + lambda_ * theta@theta


def stochastic_gradient_descent(x, y,
                                lr,
                                theta,
                                lambda_):
    # l_theta (X_ik, y_ik)
    diff_loss = diff_loss_th(x, y, theta, lambda_)
    new_theta = theta - lr * diff_loss
    return new_theta


def diff_loss_th(x, y, theta, lambda_):
    return (-y * x if 1 - (y * (x@theta)) > 0 else 0) + (2 * lambda_ * theta)

if __name__ == '__main__':
    main()
