class Dealer():
    def __init__(self, dealer_showing):
        self.cards = [dealer_showing]

    def get_value(self):
        dealer_sum = 0
        n_ace = 0

        for card in self.cards:
            if card == 1:
                n_ace += 1
            else:
                dealer_sum += card

        while True:
            if n_ace <= 0:
                break

            dealer_sum += 11
            n_ace -= 1

            if dealer_sum > 21:
                n_ace += 1
                dealer_sum -= 11
                dealer_sum += n_ace
                break

        return dealer_sum

    def hit(self, card):
        self.cards.append(card)

    def bust(self):
        return self.get_value() > 21

    def possible_hit(self):
        if self.get_value() >= 17:
            return False
        else:
            return True