import numpy as np
import argparse


def main(args):
    # hyperparameters
    size = args.size
    gamma = args.gamma
    
    target_k = [10, 100]
    actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]

    map_values = np.zeros((size, size))
    terminal_states = [[0, 0], [size - 1, size - 1]]
    states = [[i, j] for i in range(size) for j in range(size)]

    target_k.sort()
    target_k.reverse()

    iteration = 0
    while True:
        if iteration in target_k:
            target_k.pop()
            print(f"{iteration}'th iteration after")
            print(map_values)
        if len(target_k) == 0:
            break
        map_copied = np.copy(map_values)

        for state in states:
            future_rewards = 0

            for action in actions:
                next_state, reward = do_action(state, action, terminal_states)
                future_rewards += (1 / len(actions)) * (reward + (gamma * map_values[next_state[0], next_state[1]]))
            map_copied[state[0], state[1]] = future_rewards

        map_values[:] = map_copied
        iteration += 1


def do_action(state, action, terminal_states):
    if state in terminal_states:
        return (state, 0)

    reward = -1
    next_state = np.array(state) + np.array(action)
    if -1 in next_state or 4 in next_state:
        next_state = state
        
    return (next_state, reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input some arguments')
    parser.add_argument('--size', nargs='?', default=4, type=int,
                        help='an integer for the size of grid world (width=height)')
    parser.add_argument('--gamma', nargs='?', default=1, type=float,
                        help='an flaot number for the discount factor')
    print(parser.parse_args())
    main(parser.parse_args())