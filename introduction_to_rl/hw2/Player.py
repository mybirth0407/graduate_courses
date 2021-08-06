class Player:
    def __init__(self, player_sum, usable_ace, dealer_showing):
        self.player_sum = player_sum
        self.dealer_showing = dealer_showing
        self.usable_ace = usable_ace
        self.using_ace = self.usable_ace

    def get_state(self):
        return (self.player_sum, self.usable_ace, self.dealer_showing)

    def get_value(self):
        return self.player_sum

    def hit(self, card):
        if self.using_ace and self.player_sum + card > 21:
            self.using_ace = False
            self.player_sum += card - 10
        else:
            self.player_sum += card

    def bust(self):
        return self.get_value() > 21

    def possible_hit(self, policy):
        return policy[self.get_state()]

    def stick(self):
        pass