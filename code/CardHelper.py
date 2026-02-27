class CardHelper:
    SUITS = [sum(1 << i for i in range(0, 13)),
             sum(1 << i for i in range(13, 26)),
             sum(1 << i for i in range(26, 39)),
             sum(1 << i for i in range(39, 52))
    ]

    @staticmethod
    def get_cards_in_suit(suit,hand):
        return hand & CardHelper.SUITS[suit]
    @staticmethod
    def card_to_hand(card):
        # print(card)
        return 1 << card

    @staticmethod
    def list_to_hand(ls):
        hand = 0
        for card in ls:
            hand |= CardHelper.card_to_hand(card)
        return hand

    @staticmethod
    def get_card_rank(card, isHand=True):
        if isHand:
            bit = card & -card  # isolate lowest set bit
            idx = bit.bit_length() - 1  # bit position
        else:
            idx = card
        suit = idx // 13
        rank = idx % 13 + 2
        return rank

    @staticmethod
    def get_card_suit(card, isHand=True):
        if isHand:
            bit = card & -card  # isolate lowest set bit
            idx = bit.bit_length() - 1  # bit position
        else:
            idx = card
        suit = idx // 13
        rank = idx % 13 + 2
        return suit

    @staticmethod
    def get_card_rank_suit(card,isHand=True):
        if isHand:
            bit = card & -card  # isolate lowest set bit
            idx = bit.bit_length() - 1  # bit position
        else:
            idx = card
        suit = idx // 13
        rank = idx % 13 + 2
        return suit,rank
    @staticmethod
    def has_card(hand, card):
        return (hand & CardHelper.card_to_hand(card)) != 0

    @staticmethod
    def not_has_card(hand,card):
        return (hand & CardHelper.card_to_hand(card)) == 0

    @staticmethod
    def remove_hand(hand1,hand2):
        hand1 &= ~hand2
        return hand1
    @staticmethod
    def remove_card(hand, card):
        hand &= ~CardHelper.card_to_hand(card)
        return hand

    @staticmethod
    def add_card(hand, card):
        hand |= CardHelper.card_to_hand(card)
        return hand

    @staticmethod
    def get_num_cards(hand):
        return hand.bit_count()

    @staticmethod
    def get_suit_num_cards(hand,suit):
        return (hand & CardHelper.SUITS[suit]).bit_count()

    @staticmethod
    def can_follow_suit(hand,suit):
        return CardHelper.get_suit_num_cards(hand,suit) != 0

    @staticmethod
    def get_shared_cards(hand1,hand2):
        return hand1 & hand2

    @staticmethod
    def get_difference(hand1,hand2):
        return hand1 & ~hand2

    @staticmethod
    def get_lowest_card_suit(hand):
        bit = hand & -hand  # isolate lowest set bit
        idx = bit.bit_length() - 1  # bit position
        return idx

    @staticmethod
    def get_highest_card_suit(hand):
        # bit = hand & -hand  # isolate lowest set bit
        idx = hand.bit_length() - 1  # bit position
        return idx

    @staticmethod
    def to_list(hand):
        cards = []
        while hand:
            bit = hand & -hand  # isolate lowest set bit
            idx = bit.bit_length() - 1  # bit position
            cards.append(idx)
            hand ^= bit  # clear that bit
        return cards

    @staticmethod
    def print_bits(hand):
        width = 52
        print(f"{hand:0{width}b}")

    @staticmethod
    def iter_cards(hand):
        while hand:
            bit = hand & -hand     # lowest set bit
            idx = bit.bit_length() - 1
            suit = idx // 13
            rank = idx % 13
            yield suit, rank+2
            hand ^= bit

    @staticmethod
    def get_highest_card(hand):
        highrank = 0
        highsuit = 0
        for suit, rank in CardHelper.iter_cards(hand):
            if rank > highrank:
                highrank = rank
                highsuit = suit
        return CardHelper.to_card(highsuit,highrank)

    @staticmethod
    def to_card(suit,rank):
        return 13*suit + rank-2

    @staticmethod
    def hand_to_card(hand):
        bit = hand & -hand  # isolate lowest set bit
        idx = bit.bit_length() - 1
        return idx

    @staticmethod
    def str_to_card(string):
        """
            Helper function to convert string input in human interface to number for game
            """
        rankStr = string[0]
        if rankStr == 'A':
            rank = 14
        elif rankStr == 'K':
            rank = 13
        elif rankStr == 'Q':
            rank = 12
        elif rankStr == 'J':
            rank = 11
        elif rankStr == 'T':
            rank = 10
        else:
            rank = int(rankStr)
        suit = string[1]
        suits = {'C':0,'D':1,'H':2,'S':3}
        return CardHelper.to_card(suits[suit],rank)

    @staticmethod
    def get_str_suit(suit):
        return ['C', 'D', 'H', 'S'][suit]
    @staticmethod
    def to_str(card):
        suit,rank = CardHelper.get_card_rank_suit(card,False)
        # if rank == 10:
        #     rank = "T"
        if rank == 11:
            rank = "J"
        elif rank == 12:
            rank = "Q"
        elif rank == 13:
            rank = "K"
        elif rank == 14:
            rank = "A"
        suits = ['C', 'D', 'H', 'S']
        return str(rank)+suits[suit]

    @staticmethod
    def to_str_hand(hand):
        str = ""
        ls = CardHelper.to_list(hand)
        for card in ls:
            str += CardHelper.to_str(card) + " "
        return str

    @staticmethod
    def get_max_rank(hand):
        return max([rank for suit,rank in CardHelper.iter_cards(hand)])

    @staticmethod
    def get_highest_losing_card(hand,trick,trumpSuit):
        topCard = CardHelper.get_highest_card_in_trick(trick,trumpSuit)
        leadSuit = CardHelper.get_card_suit(trick[0][1], isHand=False)
        if CardHelper.get_card_suit(topCard,isHand=False) != leadSuit:
            return CardHelper.get_highest_card(hand)
        cardRank = 0
        for suit,rank in CardHelper.iter_cards(hand):
            if rank > cardRank and rank < CardHelper.get_card_rank(topCard,isHand=False):
                cardRank = rank
        if cardRank == 0:
            return -1
        else:
            return CardHelper.to_card(leadSuit,cardRank)
    @staticmethod
    def get_highest_card_in_trick(trick,trumpSuit):
        winner = trick[0][0]
        card1 = trick[0][1]
        leadSuit = CardHelper.get_card_suit(card1, isHand=False)
        for (player, card2) in trick:
            if not CardHelper.card_wins(card1,card2,leadSuit,trumpSuit):
                winner = player
                card1 = card2 + 0
        return card1

    @staticmethod
    def card_wins(card1, card2, lead_suit, trump_suit):
        s1,r1 = CardHelper.get_card_rank_suit(card1,isHand=False)
        s2,r2 = CardHelper.get_card_rank_suit(card2,isHand=False)
        # Same card
        if card1 == card2:
            return 0

        # Trump logic
        if s1 == trump_suit and s2 != trump_suit:
            return 1
        if s2 == trump_suit and s1 != trump_suit:
            return 0

        # Follow lead logic
        if s1 == lead_suit and s2 != lead_suit:
            return 1
        if s2 == lead_suit and s1 != lead_suit:
            return 0

        # Same suit → compare rank
        if s1 == s2:
            return r1 > r2

        # Neither followed lead or trump
        return 0


if __name__ == "__main__":
    print(CardHelper.card_wins(CardHelper.str_to_card("QC"),CardHelper.str_to_card("TC"),0,2))