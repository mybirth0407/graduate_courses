from MonteCarloES import MonteCarloES
import argparse
import random

def main(args):
    random.seed(args.seed)
    
    MCES = MonteCarloES()
    MCES.train(args.n_episodes)
    MCES.print_policy()
    MCES.print_value_function()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input some arguments')
    parser.add_argument('--n_episodes', nargs='?', default=500000, type=int,
                        help='an integer for the number of episodes')
    parser.add_argument('--seed', nargs='?', default=42, type=int,
                        help='an integer for the random seed')
    print(parser.parse_args())
    main(parser.parse_args())