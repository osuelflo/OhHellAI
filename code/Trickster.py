"""
CSharpBot.py
============
Python port of the C# BaseBot + OhHellBot card-play and bidding logic.
Designed to plug into the existing OhHellState / CardHelper infrastructure.

Public API (mirrors OhHellState usage):
    bot = CSharpBot(state, player)
    card = bot.suggest_next_card()   # returns a card int
    bid  = bot.suggest_bid()         # returns an int bid
"""

from CardHelper import CardHelper
from efficientISMCTS import *
from math import floor


# ---------------------------------------------------------------------------
# Thin wrappers so the port reads as close to the C# source as possible
# ---------------------------------------------------------------------------

def _rank_sort(card: int) -> int:
    """Rank sort: simply the card index (0-51). Higher = stronger within suit."""
    return card


def _effective_suit(card: int, trump_suit: int) -> int:
    """In standard Oh Hell there are no jacks-as-trump, so effective suit == real suit."""
    return CardHelper.get_card_suit(card, isHand=False)


def _is_trump(card: int, trump_suit: int) -> bool:
    return _effective_suit(card, trump_suit) == trump_suit


def _high_rank_in_suit(suit: int) -> int:
    """The highest possible card index in a suit (the Ace = rank 14 → index suit*13+12)."""
    return suit * 13 + 12   # Ace is the 13th card (index 12 within suit)


def _is_card_high(card: int, discards: int, trump_suit: int) -> bool:
    """
    Returns True if `card` is the current 'boss' in its suit —
    i.e. no card that beats it in the same suit is still live.
    """
    suit = _effective_suit(card, trump_suit)
    rank_sort_val = _rank_sort(card)
    high_rank = _high_rank_in_suit(suit)

    if rank_sort_val == high_rank:
        return True

    # Count how many cards above `card` in the same suit have already been played
    cards_above_in_deck = sum(
        1 for r in range(rank_sort_val + 1, suit * 13 + 13)   # all higher cards in suit
    )
    cards_above_played = bin(
        discards & sum(1 << r for r in range(rank_sort_val + 1, suit * 13 + 13))
    ).count('1')

    return cards_above_played == cards_above_in_deck


def _iter_legal(hand: int):
    """Yield individual card ints from a hand bitmask."""
    return CardHelper.to_list(hand)


# ---------------------------------------------------------------------------
# BaseBot helpers (ported from BaseBot<T>)
# ---------------------------------------------------------------------------

def _lowest_card_from_weakest_suit(legal_cards: list, discards: int, trump_suit: int) -> int:
    """
    Port of BaseBot.LowestCardFromWeakestSuit.
    Dumps the lowest card from the weakest (most expendable) non-trump suit.
    """
    non_trump = [c for c in legal_cards if not _is_trump(c, trump_suit)]

    if not non_trump:
        return min(legal_cards, key=_rank_sort)

    # Group by suit
    suit_groups: dict[int, list] = {}
    for c in non_trump:
        s = _effective_suit(c, trump_suit)
        suit_groups.setdefault(s, []).append(c)

    def cards_played_in_suit(suit):
        return bin(discards & CardHelper.SUITS[suit]).count('1')

    # 1. Try to ditch a non-boss singleton whose suit has the most remaining cards outstanding
    singletons = [(s, cs[0]) for s, cs in suit_groups.items()
                  if len(cs) == 1 and not _is_card_high(cs[0], discards, trump_suit)]
    if singletons:
        best = min(singletons, key=lambda sc: cards_played_in_suit(sc[0]))
        return best[1]

    # 2. Try doubletons
    doubletons = [(s, sorted(cs, key=_rank_sort)) for s, cs in suit_groups.items() if len(cs) == 2]
    doubletons.sort(key=lambda sc: cards_played_in_suit(sc[0]))
    for s, cards in doubletons:
        low, high = cards[0], cards[1]
        rank_low = CardHelper.get_card_rank(low, isHand=False)
        rank_high = CardHelper.get_card_rank(high, isHand=False)
        if _is_card_high(high, discards, trump_suit) and rank_low < rank_high - 1:
            return low
        if rank_high != 13:   # no king present
            return low

    # 3. Lowest card from longest non-trump suit
    longest = max(suit_groups.values(), key=len)
    return min(longest, key=_rank_sort)


def _try_take_em(
    legal_cards: list,
    trick: list,            # list of (player, card_int)
    discards: int,
    trump_suit: int,
    is_partner_taking: bool,
    card_taking_trick: int,
    n_players: int,
    player_hand: int,       # full hand bitmask of current player
) -> int | None:
    """
    Port of BaseBot.TryTakeEm (card-play strategy for taking tricks).
    Returns a card int, or None if no specific suggestion.

    Simplified for Oh Hell (no partnership signalling, individual game only).
    """
    suggestion = None

    if not trick:
        # ---------- We are leading ----------
        # Boss cards: play the highest-ranked boss in our longest suit
        boss_cards = [c for c in legal_cards if _is_card_high(c, discards, trump_suit)]
        if boss_cards:
            # favour boss in longest suit
            def suit_count(c):
                s = _effective_suit(c, trump_suit)
                return sum(1 for x in legal_cards if _effective_suit(x, trump_suit) == s)
            suggestion = max(boss_cards, key=suit_count)
        # (no partnership signalling in Oh Hell)
        return suggestion

    # ---------- We are following ----------
    first_card = trick[0][1]
    trick_suit = _effective_suit(first_card, trump_suit)
    n_active = n_players
    last_to_play = len(trick) == n_active - 1

    cards_in_suit = [c for c in legal_cards if _effective_suit(c, trump_suit) == trick_suit]
    has_trump = any(_is_trump(c, trump_suit) for c in legal_cards)

    if cards_in_suit:
        # We can follow suit
        trick_has_trump = any(_is_trump(c, trump_suit) for _, c in trick)
        if _is_trump(first_card, trump_suit) or not trick_has_trump:
            # Trump-led or no trump in trick yet
            if is_partner_taking and last_to_play:
                pass   # don't over-take partner who's already winning last
            elif last_to_play:
                highest_in_trick = max(
                    (_rank_sort(c) for _, c in trick
                     if _effective_suit(c, trump_suit) == trick_suit),
                    default=-1
                )
                suggestion = next(
                    (c for c in sorted(cards_in_suit, key=_rank_sort)
                     if _rank_sort(c) > highest_in_trick),
                    None
                )
            else:
                high_card = max(cards_in_suit, key=_rank_sort)
                if _is_card_high(high_card, discards, trump_suit):
                    suggestion = high_card
    elif has_trump:
        # Can't follow suit but have trump
        if is_partner_taking and last_to_play:
            pass   # don't trump over winning partner
        else:
            trick_trumps = [c for _, c in trick if _is_trump(c, trump_suit)]
            if not trick_trumps:
                # No trump yet — play lowest trump
                suggestion = min(
                    (c for c in legal_cards if _is_trump(c, trump_suit)),
                    key=_rank_sort
                )
            else:
                highest_trump_in_trick = max(_rank_sort(c) for c in trick_trumps)
                higher_trumps = [
                    c for c in legal_cards
                    if _is_trump(c, trump_suit) and _rank_sort(c) > highest_trump_in_trick
                ]
                suggestion = min(higher_trumps, key=_rank_sort) if higher_trumps else None
    # else: can't follow, no trump — fall through to dump

    return suggestion


def _try_dump_em(
    legal_cards: list,
    trick: list,            # list of (player, card_int)
    trump_suit: int,
    n_players: int,
    is_trump_fn,
    rank_sort_fn,
) -> int:
    """
    Port of OhHellBot.TryDumpEm — play to deliberately LOSE a trick.
    """
    suggestion = None
    first_card = next((c for _, c in trick), None) if trick else None

    if first_card is None:
        # Leading — play absolute lowest
        suggestion = min(legal_cards, key=rank_sort_fn)
    else:
        trick_suit = _effective_suit(first_card, trump_suit)
        cards_in_suit = [c for c in legal_cards if _effective_suit(c, trump_suit) == trick_suit]
        trick_has_trump = any(is_trump_fn(c) for _, c in trick)

        if cards_in_suit:
            if trick_has_trump and not is_trump_fn(first_card):
                # Lead suit wasn't trump but trick has trump — dump our highest
                suggestion = max(cards_in_suit, key=rank_sort_fn)
            else:
                # Dump highest card below the current winner
                trick_taker_rank = max(
                    rank_sort_fn(c) for _, c in trick
                    if _effective_suit(c, trump_suit) == trick_suit
                )
                below = [c for c in cards_in_suit if rank_sort_fn(c) < trick_taker_rank]
                suggestion = max(below, key=rank_sort_fn) if below else None
                if suggestion is None and len(trick) == n_players - 1:
                    suggestion = max(cards_in_suit, key=rank_sort_fn)
        elif trick_has_trump:
            max_trump_rank = max(
                rank_sort_fn(c) for _, c in trick if is_trump_fn(c)
            )
            below_trump = [
                c for c in legal_cards
                if is_trump_fn(c) and rank_sort_fn(c) < max_trump_rank
            ]
            non_trump = [c for c in legal_cards if not is_trump_fn(c)]
            suggestion = (
                max(below_trump, key=rank_sort_fn) if below_trump
                else (max(non_trump, key=rank_sort_fn) if non_trump else None)
            )
        else:
            non_trump = [c for c in legal_cards if not is_trump_fn(c)]
            all_sorted = sorted(legal_cards, key=rank_sort_fn)
            suggestion = (
                max(non_trump, key=rank_sort_fn) if non_trump
                else max(all_sorted, key=rank_sort_fn)
            )

    non_trump = [c for c in legal_cards if not is_trump_fn(c)]
    return (
        suggestion
        or (min(non_trump, key=rank_sort_fn) if non_trump else None)
        or min(legal_cards, key=rank_sort_fn)
    )


def _count_sure_tricks(hand_cards: list, discards: int, trump_suit: int) -> int:
    """
    Port of BaseBot.CountSureTricks.
    Counts trump tricks we will take regardless of play order.
    """
    my_trump = [c for c in hand_cards if _is_trump(c, trump_suit)]
    played_trump = [
        c for c in _iter_legal(discards) if _is_trump(c, trump_suit)
    ]
    known_trump = sorted(my_trump + played_trump, key=_rank_sort, reverse=True)

    high_rank = _high_rank_in_suit(trump_suit)
    next_rank = high_rank
    lost_tricks = 0
    sure_tricks = 0

    for t in known_trump:
        rank = _rank_sort(t)
        lost_tricks += next_rank - rank
        if t in my_trump:
            if lost_tricks <= 0:
                sure_tricks += 1
            else:
                lost_tricks -= 1
        next_rank = rank - 1

    return sure_tricks


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CSharpBot:
    """
    Drop-in card-play + bidding bot ported from the C# OhHellBot / BaseBot.

    Usage:
        bot = CSharpBot(state, player)
        card = bot.suggest_next_card()
        bid  = bot.suggest_bid()
    """

    def __init__(self, state, player: int):
        """
        Parameters
        ----------
        state  : OhHellState   (your existing Python game state)
        player : int           player index this bot is acting for
        """
        self.state = state
        self.player = player
        self.trump = state.trumpSuit

    # ------------------------------------------------------------------
    # Helpers that delegate to CardHelper / state
    # ------------------------------------------------------------------

    def _is_trump(self, card: int) -> bool:
        return _is_trump(card, self.trump)

    def _rank_sort(self, card: int) -> int:
        return _rank_sort(card)

    def _is_card_high(self, card: int) -> bool:
        return _is_card_high(card, self.state.discards, self.trump)

    def _legal_list(self) -> list:
        return _iter_legal(self.state.GetMoves())

    # ------------------------------------------------------------------
    # Public: suggest_next_card
    # ------------------------------------------------------------------

    def suggest_next_card(self) -> int:
        """
        Port of OhHellBot.SuggestNextCard.
        Returns the card int the bot wants to play.
        """
        state   = self.state
        player  = self.player
        trick   = state.currentTrick
        legal   = self._legal_list()
        discards = state.discards

        bid         = state.bids[player]
        hand_cards  = _iter_legal(state.playerHands[player])
        sure_tricks = _count_sure_tricks(hand_cards, discards, self.trump)
        tricks_taken = state.tricksTaken[player]
        tricks_needed = bid - tricks_taken

        n_players = state.numberOfPlayers

        # Detect whether a partner is currently winning (Oh Hell is individual,
        # so is_partner_taking is always False, but keep the slot for extensibility)
        is_partner_taking = False
        card_taking_trick = self._get_current_trick_winner()

        # ── 1. Already made bid: try to dump remaining cards ──
        if tricks_needed == sure_tricks:
            return _try_dump_em(
                legal, trick, self.trump, n_players,
                self._is_trump, self._rank_sort
            )

        # ── 2. Need every remaining trick: lead strongest ──
        if not trick and tricks_needed == len(hand_cards):
            trump_cards = [c for c in legal if self._is_trump(c)]
            if trump_cards:
                return max(trump_cards, key=self._rank_sort)
            boss = next((c for c in sorted(legal, key=self._rank_sort, reverse=True)
                         if self._is_card_high(c)), None)
            if boss:
                return boss
            return max(legal, key=self._rank_sort)

        # ── 3. Normal case: try to take a trick ──
        suggestion = _try_take_em(
            legal, trick, discards, self.trump,
            is_partner_taking, card_taking_trick,
            n_players, state.playerHands[player]
        )

        if suggestion is not None:
            return suggestion

        if all(self._is_trump(c) for c in legal):
            return min(legal, key=self._rank_sort)

        return _lowest_card_from_weakest_suit(legal, discards, self.trump)

    # ------------------------------------------------------------------
    # Public: suggest_bid
    # ------------------------------------------------------------------

    def suggest_bid(self) -> int:
        """
        Port of OhHellBot.SuggestBid.
        Returns an integer bid (0 … tricksInRound).
        """
        state  = self.state
        player = self.player
        hand   = state.playerHands[player]
        hand_cards = _iter_legal(hand)
        n_cards = len(hand_cards)
        n_tricks = state.tricksInRound
        trump = self.trump

        trump_cards = sorted(
            [c for c in hand_cards if self._is_trump(c)],
            key=self._rank_sort, reverse=True
        )
        off_suit_cards = [c for c in hand_cards if not self._is_trump(c)]

        # ── Estimate trump tricks ──
        est = 0.0
        n_trump = float(len(trump_cards))

        # Voids / singletons / doubletons in off-suits let us trump in
        for suit in range(4):
            if suit == trump:
                continue
            count_in_suit = sum(
                1 for c in off_suit_cards
                if CardHelper.get_card_suit(c, isHand=False) == suit
            )
            max_len = 2 + (1 if state.numberOfPlayers == 3 else 0)
            if count_in_suit < max_len:
                use = max_len - count_in_suit
                trump_in = min(n_trump, use)
                n_trump -= trump_in
                est += trump_in

        # Remaining trump cards: estimate based on gaps
        remaining_trump = trump_cards[:int(round(n_trump))]
        for card in remaining_trump:
            n_above  = sum(1 for c in remaining_trump if self._rank_sort(c) > self._rank_sort(card))
            n_below  = sum(1 for c in remaining_trump if self._rank_sort(c) < self._rank_sort(card))
            high     = _high_rank_in_suit(trump)
            n_ranks_above = high - self._rank_sort(card)
            n_gaps_above  = n_ranks_above - n_above

            if n_below >= n_gaps_above:
                est += 1.0
            elif n_below == n_gaps_above - 1:
                est += 0.75
            elif n_below == n_gaps_above - 2:
                est += 0.25

        # ── Off-suit aces ──
        off_aces = [c for c in off_suit_cards
                    if CardHelper.get_card_rank(c, isHand=False) == 14]
        for ace in off_aces:
            ace_suit = CardHelper.get_card_suit(ace, isHand=False)
            n_in_suit = sum(
                1 for c in off_suit_cards
                if CardHelper.get_card_suit(c, isHand=False) == ace_suit
            )
            max_len = n_cards // 2 + (1 if state.numberOfPlayers == 3 else 0)
            est += 1.0 if n_in_suit <= max_len else 0.5

        # ── Off-suit kings (no ace in same suit) ──
        off_kings = [
            c for c in off_suit_cards
            if CardHelper.get_card_rank(c, isHand=False) == 13
            and not any(
                CardHelper.get_card_suit(a, isHand=False) ==
                CardHelper.get_card_suit(c, isHand=False)
                for a in off_aces
            )
        ]
        for king in off_kings:
            king_suit = CardHelper.get_card_suit(king, isHand=False)
            n_in_suit = sum(
                1 for c in off_suit_cards
                if CardHelper.get_card_suit(c, isHand=False) == king_suit
            )
            if n_in_suit > 1:
                max_len = n_cards // 2 + (1 if state.numberOfPlayers == 3 else 0)
                est += 1.0 if n_in_suit <= max_len - 1 else 0.25

        target = max(0, int(floor(est)))

        # Clamp to legal range
        target = max(0, min(target, n_tricks))

        # Last-bidder constraint: bids can't sum to n_tricks
        bid_position = self._get_bid_position()
        if bid_position == state.numberOfPlayers - 1:
            current_sum = sum(state.bids)
            if current_sum + target == n_tricks:
                target = target + 1 if target + 1 <= n_tricks else target - 1

        return max(0, min(target, n_tricks))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_current_trick_winner(self) -> int:
        """Returns the card int currently winning the trick, or -1 if trick is empty."""
        trick = self.state.currentTrick
        if not trick:
            return -1
        winner_card = trick[0][1]
        lead_suit = CardHelper.get_card_suit(winner_card, isHand=False)
        for _, card in trick:
            if not self.state.trickWinnerTable[winner_card][card][lead_suit][self.trump]:
                winner_card = card
        return winner_card

    def _get_bid_position(self) -> int:
        """Returns how many players have already bid (0-indexed position of this player)."""
        # Bidding starts from the player after the dealer
        state = self.state
        p = state.GetNextPlayer(state.dealer)
        for i in range(state.numberOfPlayers):
            if p == self.player:
                return i
            p = state.GetNextPlayer(p)
        return 0


# ---------------------------------------------------------------------------
# Integration helpers: convenience wrappers for PlayRound usage
# ---------------------------------------------------------------------------

def csharp_suggest_card(state, player: int) -> int:
    """One-liner to get a card suggestion from the C# bot for `player` in `state`."""
    return CSharpBot(state, player).suggest_next_card()


def csharp_suggest_bid(state, player: int) -> int:
    """One-liner to get a bid suggestion from the C# bot for `player` in `state`."""
    return CSharpBot(state, player).suggest_bid()


# ---------------------------------------------------------------------------
# Quick sanity test (run with:  python CSharpBot.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("CSharpBot module loaded successfully.")
    print("To test, import into your OhHellState and call:")
    print("   from CSharpBot import CSharpBot")
    print("   card = CSharpBot(state, player).suggest_next_card()")
    for i in range(1000):
        mcIterations = [1000, 1000, 1000, 1000]
        dealer = 3
        numPlayers = 4
        for numTricks in range(10,11):
            main = True
            isAI = [True, True, True, True]
            useHeuristic = [True, True, True, True]
            bidStyle = ["normal", "normal", "normal", "normal"]
            enterCards = [False, False, False, False]
            state = OhHellState(numPlayers, numTricks, dealer, main, isAI, useHeuristic, bidStyle, enterCards)
            bids = ""
            for p in range(numPlayers):
                csb = CSharpBot(state,p)
                bids += str(csb.suggest_bid()) + " "
            print(bids, "---------", state.bids)
