# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2021-09-01 23:07:55
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2021-10-24 01:07:04

import numpy as np
import matplotlib.pyplot as plt


def main():
    np.seterr(invalid='ignore', over='ignore')  # suppress warning caused by division by inf

    # setup hyperparameters
    # range of x
    domain = (-5, 20)
    # step size
    learning_rate = 4
    # how many gradient descent loop
    n_epochs = 500
    # for print func, no print if `print_interval` > n_epcohs or None
    print_interval = 100
    # for just logging in implementation process, no use this variable.
    checkpoint_interval = 100
    # how many experiments
    k = 100

    domain_split = 100
    x = np.linspace(domain[0], domain[1], domain_split)
    # random starting points

    converged_points = []
    for i in range(1, k+1):
        print(f'{i}th experiment')
        starting_point = np.random.randint(domain[0], domain[1])
        converged_point, _ = gradient_descent(
            starting_point,
            lr=learning_rate,
            n_epochs=n_epochs,
            print_interval=print_interval,
            checkpoint_interval=checkpoint_interval
        )
        converged_points.append(converged_point)
        print()

    plt.title(f'learning rate: {learning_rate}')
    plt.plot(x, f(x), 'k')
    plt.plot(x, fprime(x), 'b')
    for converged_point in converged_points:
        if domain[0] <= converged_point and converged_point <= domain[1]:
            plt.plot(converged_point, f(converged_point), 'rx')
    plt.show()


def f(x):
    return 1/(1 + np.exp(3*(x-3))) * 10 * x**2  + 1 / (1 + np.exp(-3*(x-3))) * (0.5*(x-10)**2 + 50)

def fprime(x):
    return 1 / (1 + np.exp((-3)*(x-3))) * (x-10) + 1/(1 + np.exp(3*(x-3))) * 20 * x + (3* np.exp(9))/(np.exp(9-1.5*x) + np.exp(1.5*x))**2 * ((0.5*(x-10)**2 + 50) - 10 * x**2)

def print_point(*points):
    if len(points) == 1:
        points = [points]
    print(f"x:\t\t{' -> '.join(map(str, [p for p in points]))}\n"
        + f"f(x):\t\t{' -> '.join(map(str, [f(p) for p in points]))}\n"
        + f"fprime(x):\t:{' -> '.join(map(str, [fprime(p) for p in points]))}")

def gradient_descent(
    starting_point,
    lr,
    n_epochs=10000,
    print_interval=1000,
    checkpoint_interval=1000
):

    new_point = starting_point
    checkpoints = []

    print(f'initial settings\n'
        + f'learning_rate:\t{lr}\n'
        + f'x:\t\t {new_point}\n'
        + f'f(x):\t\t {f(new_point)}\n'
        + f'fprime(x):\t: {fprime(new_point)}')

    for epoch in range(1, n_epochs+1):
        # gradient descent update rule
        # theta[k+1] = theta[k] - alpha * fprime(theta[k])
        old_point = new_point
        new_point = old_point - lr * fprime(old_point)

        if print_interval is not None\
            and epoch % print_interval == 0:

            print(f'===== epoch {epoch} =====')
            print_point(old_point, new_point)

        if checkpoint_interval is not None\
            and epoch % checkpoint_interval == 0:

            checkpoints.append(new_point)

    # To logging I use checkpoints. but no use at submission version.
    return new_point, checkpoints


if __name__ == '__main__':
    main()
