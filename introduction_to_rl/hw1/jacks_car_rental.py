import numpy as np
import random
import argparse
import math
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

class JacksCarRental:
    def __init__(self, args):
        self.max_cars = args.max_cars
        self.rental_cost = args.rental_cost
        self.move_cost = args.move_cost
        self.max_move = args.max_move
        self.upper_bound = args.upper_bound
        self.gamma = args.gamma
        self.avg_requests = args.avg_requests
        self.avg_returns = args.avg_returns

        # calculate time for possion distribution is optimized by caching poisson distribution.
        self.poissons = {2: {}, 3: {}, 4: {}}
        for i in range(2, 5):
            for j in range(self.upper_bound):
                self.poissons[i][j] = poisson.pmf(j, i)

        self.states = [[i, j] for i in range(self.max_cars + 1) for j in range(self.max_cars + 1)]
        self.actions = np.arange(-self.max_move, self.max_move + 1)
        self.policy = np.zeros((self.max_cars + 1, self.max_cars + 1))
        self.values = np.zeros((self.max_cars + 1, self.max_cars + 1))

    def get_gain(self, state, action):
        gain = 0.0
        gain -= self.move_cost * abs(action)

        # print(state, action)
        for first in range(self.upper_bound):
            for second in range(self.upper_bound):
                orders = [int(min(state[0] - action, self.max_cars)),
                          int(min(state[1] + action, self.max_cars))]

                rental_orders = [min(orders[0], first),
                                 min(orders[1], second)]
                r = sum(rental_orders) * self.rental_cost
                # orders = list(np.array(orders) - np.array(rental_orders))
                orders[0] -= rental_orders[0]
                orders[1] -= rental_orders[1]
                p = self.poissons[3][first] * self.poissons[4][second]

                fix_orders = [
                    int(min(orders[0] + self.avg_returns[0], self.max_cars)),
                    int(min(orders[1] + self.avg_returns[1], self.max_cars))
                ]
                gain += p * (r + self.gamma * self.values[fix_orders[0], fix_orders[1]])
        return gain

    def policy_iteration(self, target_policy=[], target_value=[]):
        new_values = np.zeros((self.max_cars + 1, self.max_cars + 1))
        target_policy.sort()
        target_policy.reverse()
        target_value.sort()
        target_value.reverse()

        n_improved = 0
        improved = False

        while True:
            if n_improved in target_policy:
                k = target_policy.pop()
                self.print_policy(k)

            if n_improved in target_value:
                k = target_value.pop()
                self.print_value(k)

            if len(target_policy) + len(target_value) == 0:
                break

            if improved == True:
                new_policy = np.zeros((self.max_cars + 1, self.max_cars + 1))

                for first, second in self.states:
                    actaion_gains = []
                    for action in self.actions:
                        if (action >= 0 and first >= action)\
                            or (action < 0 and second >= -action):

                            actaion_gains.append(self.get_gain([first, second], action))
                        else:
                            actaion_gains.append(-float('inf'))

                    new_policy[first, second] = self.actions[np.argmax(actaion_gains)]

                self.policy = new_policy
                improved = False

                n_improved += 1
                print(f'n_improved: {n_improved}')

            for i, j in self.states:
                new_values[i, j] = self.get_gain([i, j], self.policy[i, j])

            if np.sum(np.abs(new_values - self.values)) < 0.01:
                self.values[:] = new_values
                improved = True
                continue
            self.values[:] = new_values

    def print_policy(self, k):
        fig = plt.figure(f'{k}th Policy')
        ticks = [0] + [''] * (self.max_cars - 1) + [self.max_cars]
        ax = sns.heatmap(
            self.policy.astype(int), square=True,
            xticklabels=ticks, yticklabels=ticks
        )
        ax.set_title(f'{k}th Policy')
        ax.set_xlabel('# cars at second location')
        ax.set_ylabel('# cars at first location')
        ax.invert_yaxis()
        plt.show()

    def print_value(self, k):
        fig = plt.figure(f'{k}th Value')
        ax = fig.add_subplot(111, projection='3d')
        x = []
        y = []
        for i in range(self.max_cars + 1):
            for j in range(self.max_cars + 1):
                x.append(i)
                y.append(j)
        v = []
        for i, j in self.states:
            v.append(self.values[j, i]) # second, first
        ax.scatter(x, y, v)
        ax.set_title(f'{k}th Value')
        ax.set_xlabel('# cars at second location')
        ax.set_ylabel('# cars at first location')
        ax.set_zlabel('V')
        plt.show()


def main(args):
    JacksCarRental = JacksCarRental(args)
    JacksCarRental.policy_iteration(target_policy=[4, 9], target_value=[9])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input some arguments.')
    # state
    parser.add_argument('--max_cars', nargs='?', type=int, default=20,
                        help='an integer for the max cars')
    # related to rewards
    parser.add_argument('--rental_cost', nargs='?', type=int, default=10,
                        help='an integer for the rental cost')
    parser.add_argument('--move_cost', nargs='?', type=int, default=2,
                        help='an integer for the moving cost')
    parser.add_argument('--max_move', nargs='?', type=int, default=5,
                        help='an integer for the max number of car moving')
    parser.add_argument('--upper_bound', nargs='?', type=int, default=11,
                        help='an integer for the poisson upper bound')
    parser.add_argument('--gamma', nargs='?', type=float, default=0.9,
                        help='an integer for the discount factor')
    parser.add_argument('--avg_requests', nargs='*', default=[3, 4],
                        help='an list for the average request [first, second]')
    parser.add_argument('--avg_returns', nargs='*', default=[3, 2],
                        help='an list for the average return [first, second]')
    print(parser.parse_args())
    main(parser.parse_args())