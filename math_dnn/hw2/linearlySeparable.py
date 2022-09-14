# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2021-09-14 04:27:45
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2021-09-15 15:28:55


import numpy as np
import matplotlib.pyplot as plt


def main():
    # data given from hw2
    N = 30
    np.random.seed(0)
    X = np.random.randn(2, N)
    y = np.sign(X[0,:]**2 + X[1,:]**2 - 0.7)
    theta = 0.5
    c, s = np.cos(theta), np.sin(theta)
    X = np.array([[c, -s], [s, c]])@X
    X = X + np.array([[1], [1]])

    # to easy implementation
    X = X.T

    # hyperparameters
    K = 20000
    alpha = 0.007

    # optimization target parameter
    w = np.random.normal(loc=0.0, scale=4.0, size=5)

    # observe (by plotting) that the data is not linearly separable.
    for i in range(X.shape[0]):
        if y[i] == 1:
            plt.scatter(X[i][0], X[i][1], c="red")
        else:
            plt.scatter(X[i][0], X[i][1], c="green")

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.legend()
    # plt.show()
    plt.savefig('observeX.png')

    X = transformation(X)

    for k in range(K):
        # generates a random sample from X with uniform distribution
        i_k = np.random.choice(np.arange(N), replace=True)
        loss = loss_th(X[i_k], y[i_k], w)
        w = stochastic_gradient_descent(
            X[i_k], y[i_k],
            lr=alpha,
            weight=w
        )

    # visualize the decision boundary given by
    # w[0] + w[1]*x[0]+ w[2]* (x**2) + w[3]*y + w[4]*(y**2)
    xx = np.linspace(-4, 4, 1024)
    yy = np.linspace(-4, 4, 1024)
    xx, yy = np.meshgrid(xx, yy)
    Z = w[0] + (w[1]*xx + w[2]*xx**2) + (w[3]*yy + w[4]*yy**2)
    plt.contour(xx, yy, Z, 0)

    plt.legend()
    # plt.show()
    plt.savefig('decisionBoundaryLinearlySeparable.png')


def transformation(X):
    return np.array([[1, u, u**2, v, v**2] for u, v in X])


def loss_th(x, y, weight):
    return np.log(1 + np.exp(-y * x@weight))


def diff_loss_th(x, y, weight):
    return (-y * x) / (1 + np.exp(y * x@weight))


def stochastic_gradient_descent(x, y, lr, weight):
    # l_theta (X_ik, y_ik)
    diff_loss = diff_loss_th(x, y, weight)
    new_theta = weight - lr * diff_loss
    return new_theta


if __name__ == '__main__':
    main()
