# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2021-09-12 04:15:09
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2021-09-14 23:35:21


import numpy as np
import matplotlib.pyplot as plt


def main():
    K = 10000
    alpha = 0.007
    N, p = 30, 50
    np.random.seed(0)
    a0 = np.random.normal(loc=0.0, scale=4.0, size=p)
    b0 = np.random.normal(loc=0.0, scale=4.0, size=p)
    u0 = np.random.normal(loc=0, scale=0.05, size=p)
    theta = np.concatenate((a0, b0, u0))

    X = np.random.normal(loc=0.0, scale=1.0, size=N)
    Y = f_true(X)

    xx = np.linspace(-2, 2, 1024)
    plt.plot(X, f_true(X), 'rx', label='Data points')
    plt.plot(xx, f_true(xx), 'r', label='True Fn')

    for k in range(K):
        # generates a random sample from X with uniform distribution
        i_k = np.random.choice(np.arange(N), replace=True)
        theta = stochastic_gradient_descent(
            X[i_k], Y[i_k],
            lr=alpha,
            theta=theta,
            p=p
        )
        if (k+1) % 2000 == 0:
            plt.plot(
                xx, f_th(theta, xx, p),
                label=f'Learned Fn after {k+1} iterations'
            )

    plt.legend()
    # plt.show()
    plt.savefig('twolayerSGD.png')


def f_true(x):
    return (x-2) * np.cos(4*x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def f_th(theta, x, p):
    return np.sum(
        theta[2*p:3*p] * sigmoid(
            theta[0:p] * np.reshape(x, (-1, 1)) + theta[p:2*p]
        ), axis=1
    )


def diff_f_th(theta, x, p):
    partial_a = np.sum(
        np.diag(sigmoid_prime(theta[0:p]*x + theta[p:2*p])) * theta[2*p:3*p]*x,
        axis=1
    )
    partial_b = np.sum(
        np.diag(sigmoid_prime(theta[0:p]*x + theta[p:2*p])) * theta[2*p:3*p],
        axis=1
    )
    partial_u = sigmoid(theta[0:p]*x + theta[p:2*p])

    return partial_a, partial_b, partial_u


def diff_loss_th(x, y, theta, p):
    partial_a, partial_b, partial_u = diff_f_th(theta, x, p)
    partial_a_loss = f_th(theta, x, p)*partial_a - y*partial_a
    partial_b_loss = f_th(theta, x, p)*partial_b - y*partial_b
    partial_u_loss = f_th(theta, x, p)*partial_u - y*partial_u

    return np.concatenate(
        (partial_a_loss, partial_b_loss, partial_u_loss)
    )


def stochastic_gradient_descent(x, y,
                                lr,
                                theta,
                                p):
    # l_theta (X_ik, y_ik)
    diff_loss = diff_loss_th(x, y, theta, p)
    new_theta = theta - lr * diff_loss
    return new_theta


if __name__ == '__main__':
    main()
