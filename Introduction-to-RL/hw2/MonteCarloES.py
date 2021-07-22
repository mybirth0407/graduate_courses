import random
from Environment import Environment
from Player import Player
from Dealer import Dealer
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D


class MonteCarloES:
    """
    Monte Carlo ES (Exploring Starts)
    """

    def __init__(self):
        self.epsilon = 0.1
        self.gamma = 1
        self.existed_pair = set()

        self.policy = dict()  # policy
        self.Q = dict()  # approximate state-action value
        self.returns = dict() # list of return from end of episode
        self.init()

    def init(self):
        # generate all possible states
        # Usable ace
        for player_sum in range(11, 22):
            for usable_ace in [True, False]:
                for dealer_showing in range(1, 11):
                    player_state = (player_sum, usable_ace, dealer_showing)
                    self.Q[(player_state, False)] = (0.0, 0)
                    self.Q[(player_state, True)] = (0.0, 0)
                    self.returns[(player_state, False)] = 0
                    self.returns[(player_state, True)] = 0

                    if player_sum == 20 or player_sum == 21:
                        self.policy[player_state] = False
                    else:
                        self.policy[player_state] = True

    def update(self, state_action_pair, reward):
        for sa_pair in state_action_pair:
            self.returns[sa_pair] += 1
            avg_returns, visited = self.Q[(sa_pair)]
            # self.Q[sa_pair] += avg_returns
            self.Q[sa_pair] = (
                (avg_returns * visited + reward) / (visited + 1),
                visited + 1
            )
            state = sa_pair[0]
            self.policy[state] = \
                self.Q[(state, True)][0] > self.Q[(state, False)][0]

    def generate_episode(self):
        def _new_card():
            # 1 ~ 10, j ~ k
            card_list = list(range(1, 11)) + [10, 10, 10]
            card = random.choice(card_list)
            return card

        def _add_sa_pair(state_action_pair, existed_pair, player, possible_hit):
            player_state = player.get_state()
            if player_state not in existed_pair:
                state_action_pair.append((player_state, possible_hit))
                existed_pair.add((player_state, possible_hit))

            return state_action_pair, existed_pair

        # random first state and action
        player_sum = random.randint(11, 21)
        usable_ace = random.choice([False, True])
        dealer_showing = random.randint(1, 10)

        player = Player(player_sum, usable_ace, dealer_showing)
        dealer = Dealer(dealer_showing)

        state_action_pair = []
        existed_pair = set()
        random_hit = random.choice([False, True])

        state_action_pair, existed_pair = _add_sa_pair(
            state_action_pair, existed_pair, player, random_hit
        )

        # hit
        if random_hit == True:
            player.hit(_new_card())

            while not player.bust() \
                and player.possible_hit(self.policy):

                state_action_pair, existed_pair = _add_sa_pair(
                    state_action_pair, existed_pair, player, True
                )
                player.hit(_new_card())

        # dealer win
        if player.bust():
            self.update(state_action_pair, -1)
        else:
            state_action_pair, existed_pair = _add_sa_pair(
                state_action_pair, existed_pair, player, False
            )
            dealer.hit(_new_card())

            while not dealer.bust() and dealer.possible_hit():
                dealer.cards.append(_new_card())

            # player win
            if dealer.bust() or dealer.get_value() < player.get_value():
                self.update(state_action_pair, 1)
            # dealer win
            elif dealer.get_value() > player.get_value():
                self.update(state_action_pair, -1)
            # draw
            elif dealer.get_value() == player.get_value():
                self.update(state_action_pair, 0)

    def train(self, n_episodes):
        for i in tqdm(range(n_episodes)):
            # if i % 100000 == 0:
                # self.diff_optimal(i)

            self.generate_episode()

    def print_policy(self):
        x11 = []
        y11 = []
        x12 = []
        y12 = []
        x21 = []
        y21 = []
        x22 = []
        y22 = []

        for player_state in self.policy.keys():
            # usable ace
            if player_state[1]:
                if self.policy[player_state]:
                    # dealer showing
                    x11.append(player_state[2] - 1)
                    # player sum
                    y11.append(player_state[0] - 11)
                else:
                    x12.append(player_state[2] - 1)
                    y12.append(player_state[0] - 11)
            # no usable ace
            else:
                if self.policy[player_state]:
                    x21.append(player_state[2] - 1)
                    y21.append(player_state[0] - 11)
                else:
                    x22.append(player_state[2] - 1)
                    y22.append(player_state[0] - 11)

        plt.figure(0)
        plt.title('Usable Ace')
        plt.scatter(x11, y11, color='red')
        plt.scatter(x12, y12, color='blue')
        plt.xticks(
            range(10),
            ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        )
        plt.yticks(
            range(11),
            ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        )

        plt.figure(1)
        plt.title('No Usable Ace')
        plt.scatter(x21, y21, color='red')
        plt.scatter(x22, y22, color='blue')
        plt.xticks(
            range(10),
            ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        )
        plt.yticks(
            range(11),
            ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        )

        plt.show()

    def print_value_function(self):
        import numpy as np

        def chunk(target, size):
            return list(map(
                lambda x: target[x * size:x * size + size],
                list(range(int(np.ceil(len(target) / size))))
            ))

        x1 = []
        y1 = []
        z1 = []
        x2 = []
        y2 = []
        z2 = []

        for player_state in self.policy.keys():
            if player_state[0] <= 11:
                continue

            # usable ace
            if player_state[1]:
                # dealer showing
                x1.append(player_state[2] - 1)
                # player sum
                y1.append(player_state[0] - 12)
                z1.append(self.Q[(player_state, self.policy[player_state])][0])
            # no usable ace
            else:
                # dealer showing
                x2.append(player_state[2] - 1)
                # player sum
                y2.append(player_state[0] - 12)
                z2.append(self.Q[(player_state, self.policy[player_state])][0])

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plt.title('Usable Ace')
        plt.xlabel('dealer showing')
        plt.ylabel('player sum')
        plt.xticks(
            range(10),
            ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        )
        plt.yticks(
            range(10),
            ['12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        )
        ax.set_zlim(-1, 1)

        x1 = np.array(chunk(x1, 10))
        y1 = np.array(chunk(y1, 10))
        z1 = np.array(chunk(z1, 10))
        ax.plot_surface(x1, y1, z1)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax = fig.add_subplot(122, projection='3d')
        plt.title('Usable Ace')
        plt.xlabel('dealer showing')
        plt.ylabel('player sum')
        plt.xticks(
            range(10),
            ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        )
        plt.yticks(
            range(10),
            ['12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        )
        ax.set_zlim(-1, 1)

        x2 = np.array(chunk(x2, 10))
        y2 = np.array(chunk(y2, 10))
        z2 = np.array(chunk(z2, 10))
        ax.plot_surface(x2, y2, z2)

        plt.show()
