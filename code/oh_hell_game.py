"""
oh_hell_game.py
===============
Server-optimized Oh Hell game logic with ISMCTS AI.
Stripped of print statements and terminal-focused code.
Keeps all core AI logic intact.

This is a clean version of efficientISMCTS.py designed for the web server.
"""

from CardHelper import CardHelper
import numpy as np
import time
from math import *
import random
from copy import deepcopy
import pandas as pd

class Node:
    """A node in the ISMCTS game tree."""
    
    def __init__(self, move=None, parent=None, playerJustMoved=None):
        self.move = move
        self.parentNode = parent
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.avails = 1
        self.playerJustMoved = playerJustMoved
        self.triedMoves = []

    def GetUntriedMoves(self, legalMoves):
        """Return legal moves for which this node does not have children."""
        return [
            CardHelper.to_card(suit, rank) 
            for suit, rank in CardHelper.iter_cards(legalMoves) 
            if CardHelper.not_has_card(CardHelper.list_to_hand(self.triedMoves), CardHelper.to_card(suit, rank))
        ]

    def UCBSelectChild(self, legalMoves, exploration=0.7):
        """Use UCB1 formula to select a child node."""
        legalChildren = [child for child in self.childNodes if CardHelper.has_card(legalMoves, child.move)]
        
        s = max(
            legalChildren,
            key=lambda c: float(c.wins) / float(c.visits) + exploration * sqrt(log(c.avails) / float(c.visits))
        )
        
        for child in legalChildren:
            child.avails += 1
        
        return s

    def AddChild(self, m, p):
        """Add a new child node for move m."""
        n = Node(move=m, parent=self, playerJustMoved=p)
        self.childNodes.append(n)
        self.triedMoves.append(m)
        return n

    def Update(self, terminalState, score):
        """Update node statistics after a simulation."""
        self.visits += 1
        if self.playerJustMoved is not None:
            self.wins += score


def ISMCTS(rootstate, itermax, randomRollout, mainPlayer, verbose=False):
    """
    Conduct ISMCTS search for itermax iterations.
    Returns the best move from the rootstate.
    """
    rootnode = Node()

    for i in range(itermax):
        node = rootnode
        state = rootstate.CloneAndRandomize(rootstate.playerToMove)
        
        # Select
        while state.GetMoves() != 0 and node.GetUntriedMoves(state.GetMoves()) == []:
            node = node.UCBSelectChild(state.GetMoves())
            state.DoMove(node.move)
        
        # Expand
        untriedMoves = node.GetUntriedMoves(state.GetMoves())
        if untriedMoves != []:
            m = random.choice(untriedMoves)
            player = state.playerToMove
            state.DoMove(m)
            node = node.AddChild(m, player)
        
        # Simulate
        if randomRollout:
            score = state.randomRollout(mainPlayer)
        else:
            while state.GetMoves() != 0:
                state.DoMove(random.choice(CardHelper.to_list(state.GetMoves())))
            score = state.GetScore(mainPlayer)
        
        # Backpropagate
        while node != None:
            node.Update(state, score)
            node = node.parentNode
    
    return max(rootnode.childNodes, key=lambda c: c.visits).move


class OhHellState():
    """Oh Hell game state with ISMCTS AI."""
    
    def __init__(self, n, numTricks, dealer, main, isAI, useHeuristic, bidStyle, enterCards, start=True):
        """
        Initialize Oh Hell game state.
        
        n: number of players
        numTricks: number of tricks in this round
        dealer: dealer index
        main: whether this is the main state (vs clone)
        isAI: list of booleans indicating AI players
        useHeuristic: list of booleans for particle filtering
        bidStyle: list of bid styles ('normal', 'aggressive', 'passive')
        enterCards: list of booleans for manual card entry
        start: whether to deal cards and initialize
        """
        self.deck = self.GetCardDeck()
        self.enterCards = enterCards
        self.bidStyle = bidStyle
        self.isAI = isAI
        self.main = main
        self.useHeuristic = useHeuristic
        self.players = list(range(0, n))
        self.dealer = dealer
        self.flippedOverCard = None
        self.numberOfPlayers = n
        self.playerToMove = 0
        self.tricksInRound = numTricks
        self.playerHands = [0 for p in range(0, self.numberOfPlayers)]
        self.discards = 0
        self.currentTrick = []
        self.trumpSuit = None
        self.tricksTaken = []
        self.bids = []
        self.voids = self.createVoids()
        self.haventBid = [True]*self.numberOfPlayers
        if start:
            self.trickWinnerTable = self.createTrickWinnerLookupTable()
            self.sideSuitProbs, self.trumpSuitProbs = self.getProbTables()
            self.sideOneTrickProbs, self.trumpOneTrickProbs = self.getOneTrickProbsTable()
            self.Deal()
            
            # Initialize bids to -1 (not yet placed) - using -1 instead of None so sum() works
            self.bids = [-1 for _ in range(self.numberOfPlayers)]
            
            # Set player to move to first bidder (after dealer)
            self.playerToMove = self.GetNextPlayer(self.dealer)
            
            # Only initialize probability tables if we have more than 1 trick
            if self.tricksInRound != 1:
                self.probTables = self.initializeProbTables()
                # Don't adjust probs for bids yet - no bids have been made!
        
        if main:
            self.originalHands = deepcopy(self.playerHands)

    def GetNextPlayer(self, p):
        """ Return the player to the left of the specified player
        """
        if p == self.numberOfPlayers - 1:
            next = 0
        else:
            next = p + 1
        return next

    def compressFactor(self, factor):
        """Compress factor into reasonable range."""
        import math
        x_min, x_max = 0.05, 15
        y_min, y_max = 0.5, 2
        log_x_min = math.log(x_min)
        log_x_max = math.log(x_max)
        a = (math.log(y_max) - math.log(y_min)) / (log_x_max - log_x_min)
        b = math.log(y_min) - a * log_x_min
        return math.exp(a * math.log(factor) + b)

    def adjustProbabilities(self, observer, player, card, change, probTables=None):
        """Adjust probability tables after observing an action."""
        if probTables == None:
            sum_pi = sum([self.probTables[observer][i][card] for i in range(self.numberOfPlayers + 1) if i != player])
            if sum_pi > 1e-8:
                for i in range(self.numberOfPlayers + 1):
                    if i != player and i != observer:
                        self.probTables[observer][i][card] += -change * self.probTables[observer][i][card] / sum_pi
        else:
            sum_pi = sum([probTables[observer][i][card] for i in range(self.numberOfPlayers + 1) if i != player])
            if sum_pi > 1e-8:
                for i in range(self.numberOfPlayers + 1):
                    if i != player and i != observer:
                        probTables[observer][i][card] += -change * probTables[observer][i][card] / sum_pi
            return probTables

    def getSumProbs(self, observer, probTables):
        """Get sum of probabilities for each player."""
        return [sum(probTables[observer][i]) for i in range(self.numberOfPlayers + 1)]

    def probChange(self, factor, prob):
        """Calculate probability change based on factor."""
        return 1 / (1 + exp(-(log(prob / (1 - prob)) + factor))) - prob

    def adjustProbsBids(self):
        """
        Adjust probability tables based on bidding behavior.
        Should be called AFTER all bids have been placed.
        """
        # Only adjust if we're using heuristics and all bids are in
        if not all(b is not None and b >= 0 for b in self.bids):
            return  # Not all bids placed yet
        
        p = self.GetNextPlayer(self.dealer)
        playOrder = [p]
        while p != self.dealer:
            p = self.GetNextPlayer(p)
            playOrder.append(p)
        
        for observer in range(self.numberOfPlayers):
            if self.isAI[observer] and self.useHeuristic[observer]:
                for p in range(self.numberOfPlayers):
                    player = playOrder[p]
                    if player != observer:
                        bid = self.bids[playOrder[p]]
                        expBid = (self.tricksInRound - sum(self.bids[:(p)])) / (self.numberOfPlayers - p)
                        if expBid <= 0:
                            expBid = 1 / (self.tricksInRound / self.numberOfPlayers) * 2
                        expBidFactor = (bid / expBid)
                        
                        for card in range(52):
                            if not CardHelper.has_card(self.playerHands[observer], card) and card != self.flippedOverCard:
                                rank = CardHelper.get_card_rank(card, isHand=False)
                                suit = CardHelper.get_card_suit(card, isHand=False)
                                trumpFactor = 1.5 if suit == self.trumpSuit else 1
                                
                                if rank > 13:
                                    rankFactor = 4
                                elif rank > 12:
                                    rankFactor = 3
                                elif rank > 10:
                                    rankFactor = 1.5
                                elif rank > 5:
                                    rankFactor = 1
                                elif rank > 3:
                                    rankFactor = 0.5
                                elif rank > 2:
                                    rankFactor = 0.25
                                elif rank == 2:
                                    rankFactor = 0.2
                                
                                observerSuits = CardHelper.get_suit_num_cards(self.playerHands[observer], suit)
                                suitFactor = 1
                                if observerSuits > 1.5 * self.tricksInRound / 4 and expBidFactor > 1:
                                    suitFactor = 1 / (expBidFactor * (observerSuits / (1.5 * self.tricksInRound / 4))) * 1.2
                                if expBidFactor == 0:
                                    expBidFactor = 0.75 / expBid
                                
                                if expBidFactor > 1 and rank >= 8:
                                    totalFactor = (expBidFactor * rankFactor * suitFactor * trumpFactor)
                                elif expBidFactor > 1 and rank < 8:
                                    totalFactor = (rankFactor * suitFactor / expBidFactor * trumpFactor)
                                elif expBidFactor < 1 and rank < 8:
                                    totalFactor = suitFactor / expBidFactor / rankFactor / trumpFactor
                                elif expBidFactor < 1 and rank >= 8:
                                    totalFactor = (expBidFactor / rankFactor * suitFactor / trumpFactor)
                                else:
                                    totalFactor = 1
                                
                                if totalFactor <= 0:
                                    totalFactor = 1
                                
                                totalFactor = self.compressFactor(totalFactor)
                                change = self.probChange(totalFactor, self.probTables[observer][player][card])
                                self.probTables[observer][player][card] += change
                                self.adjustProbabilities(observer, player, card, change)

    def initializeProbTables(self):
        """Initialize probability tables for particle filtering."""
        probCardDealt = (self.numberOfPlayers - 1) * self.tricksInRound / (51 - self.tricksInRound)
        probCardDealtPlayer = probCardDealt / (self.numberOfPlayers - 1)
        probCardNotDealt = 1 - probCardDealt
        probTables = []
        
        for player in range(self.numberOfPlayers):
            playerHand = self.playerHands[player]
            if self.isAI[player]:
                probTables.append([[0 for i in range(52)] for j in range(self.numberOfPlayers + 1)])
                for p in range(self.numberOfPlayers + 1):
                    for card in range(52):
                        if card == self.flippedOverCard:
                            probTables[player][p][card] = 0
                        elif p == player:
                            probTables[player][p][card] = int(CardHelper.has_card(playerHand, card))
                        else:
                            if CardHelper.has_card(playerHand, card):
                                probTables[player][p][card] = 0
                            else:
                                if p == self.numberOfPlayers:
                                    probTables[player][p][card] = probCardNotDealt
                                else:
                                    probTables[player][p][card] = probCardDealtPlayer
            else:
                probTables.append([[0 for i in range(52)] for j in range(self.numberOfPlayers + 1)])
        return probTables

    def randomRollout(self, player):
        """
        Fast rollout strategy for ISMCTS simulation.
        Uses simple heuristics rather than full ISMCTS.
        """
        while self.GetMoves() != 0:
            tricksLeftPlayer = CardHelper.get_num_cards(self.playerHands[self.playerToMove])
            tricksNeededPlayer = self.bids[player] - self.tricksTaken[player]
            playerTM = self.playerToMove
            tricksLeft = self.tricksInRound - CardHelper.get_num_cards(self.playerHands[self.playerToMove])
            tricksNeeded = self.bids[playerTM] - self.tricksTaken[playerTM]
            
            if tricksNeededPlayer > tricksLeftPlayer or tricksNeededPlayer < 0:
                return self.GetScore(player)
            
            moves = self.GetMoves()
            if len(self.currentTrick) > 0:
                card1 = self.currentTrick[0][1]
                leadSuit = CardHelper.get_card_suit(card1, isHand=False)
            
            if CardHelper.get_num_cards(moves) == 1:
                self.DoMove(CardHelper.hand_to_card(moves))
            else:
                if tricksNeeded > 0:
                    if len(self.currentTrick) == 0:
                        self.DoMove(CardHelper.get_highest_card(moves))
                    elif CardHelper.get_suit_num_cards(moves, leadSuit) != 0:
                        move = CardHelper.get_highest_card_suit(moves)
                        if self.checkWinTrick(move):
                            self.DoMove(move)
                        else:
                            self.DoMove(CardHelper.get_lowest_card_suit(moves))
                    else:
                        trumpCards = CardHelper.get_cards_in_suit(self.trumpSuit, moves)
                        if trumpCards != 0:
                            move = CardHelper.get_highest_card_suit(trumpCards)
                            if self.checkWinTrick(move):
                                self.DoMove(move)
                            else:
                                self.DoMove(CardHelper.get_lowest_card_suit(trumpCards))
                        else:
                            self.DoMove(CardHelper.get_lowest_card_suit(moves))
                else:
                    self.DoMove(CardHelper.get_lowest_card_suit(moves))
        
        return self.GetScore(player)

    def checkWinTrick(self, card2):
        """Check if card2 would win the current trick."""
        winner = self.currentTrick[0][0]
        card = self.currentTrick[0][1]
        leadSuit = CardHelper.get_card_suit(card, isHand=False)
        
        for (player, card1) in self.currentTrick:
            if self.trickWinnerTable[card1][card2][leadSuit][self.trumpSuit]:
                return False
        return True

    def getWinnerMidTrick(self):
        """Get the current winner of the trick in progress."""
        card1 = self.currentTrick[0][1]
        winner = self.currentTrick[0][0]
        leadSuit = CardHelper.get_card_suit(card1, isHand=False)
        
        for (player, card2) in self.currentTrick:
            if not self.trickWinnerTable[card1][card2][leadSuit][self.trumpSuit]:
                winner = player
                card1 = card2
        
        return card1, winner

    def createTrickWinnerLookupTable(self):
        """Pre-compute all trick winner outcomes."""
        WINNERS = np.zeros((52, 52, 4, 4))
        for c1 in range(52):
            for c2 in range(52):
                for lead in range(4):
                    for trump in range(4):
                        WINNERS[c1][c2][lead][trump] = CardHelper.card_wins(c1, c2, lead, trump)
        return WINNERS

    def createVoids(self):
        """Initialize void tracking for each player."""
        voids = {
            0: [False for p in range(0, self.numberOfPlayers)],
            1: [False for p in range(0, self.numberOfPlayers)],
            2: [False for p in range(0, self.numberOfPlayers)],
            3: [False for p in range(0, self.numberOfPlayers)]
        }
        return voids

    def getSeenCards(self, observer):
        """Returns all cards that the observer has seen."""
        return (
            self.discards | 
            self.playerHands[observer] | 
            CardHelper.list_to_hand(c for (player, c) in self.currentTrick) | 
            CardHelper.card_to_hand(self.flippedOverCard)
        )

    def getUnseenCards(self, observer):
        """Gets all cards that the observer has not seen."""
        return CardHelper.get_difference(self.deck, self.getSeenCards(observer))

    def getOneTrickProbsTable(self):
        """Returns probability tables for one-trick games."""
        return (
            pd.read_csv("probabilityData/sideOneTrickProbs.csv"),
            pd.read_csv("probabilityData/trumpOneTrickProbs.csv")
        )

    def getProbTables(self):
        """Gets probability tables for current game configuration."""
        try:
            df = pd.read_csv(f"probabilityData/{self.tricksInRound}Tricks{self.numberOfPlayers}PSideSuit.csv")
            df2 = pd.read_csv(f"probabilityData/{self.tricksInRound}Tricks{self.numberOfPlayers}PTrumpSuit.csv")
        except:
            df = pd.DataFrame()
            df2 = pd.DataFrame()
        return df, df2

    def Clone(self):
        """Create a deep clone of this game state."""
        st = OhHellState(
            self.numberOfPlayers, self.tricksInRound, self.dealer,
            False, self.isAI, self.useHeuristic, self.bidStyle, 
            self.enterCards, False
        )
        st.trickWinnerTable = self.trickWinnerTable
        st.deck = self.deck
        st.enterCards = self.enterCards
        st.isAI = self.isAI
        st.bidStyle = self.bidStyle
        st.useHeuristic = self.useHeuristic
        st.main = False
        st.players = self.players
        st.dealer = self.dealer
        st.flippedOverCard = self.flippedOverCard
        st.playerToMove = self.playerToMove
        st.trickInRound = self.tricksInRound
        st.playerHands = deepcopy(self.playerHands)
        st.discards = deepcopy(self.discards)
        st.currentTrick = deepcopy(self.currentTrick)
        st.trumpSuit = self.trumpSuit
        st.tricksTaken = deepcopy(self.tricksTaken)
        st.bids = deepcopy(self.bids)
        st.voids = deepcopy(self.voids)
        st.originalHands = deepcopy(self.originalHands)
        return st

    def randomDeal(self, observer, probTables):
        """Deal cards randomly based on probability tables."""
        unseenCards = self.getUnseenCards(observer)
        lengths = [CardHelper.get_num_cards(self.playerHands[i]) for i in range(self.numberOfPlayers)]
        players = self.players + [self.numberOfPlayers]
        players.remove(observer)
        
        obsHand = self.playerHands[observer]
        hands = deepcopy(self.playerHands)
        
        timeout_start = time.perf_counter()
        max_timeout = 2.0  # 2 second timeout
        
        while True:
            if time.perf_counter() - timeout_start > max_timeout:
                # Timeout - just do uniform random deal
                listUnseenCards = CardHelper.to_list(unseenCards)
                random.shuffle(listUnseenCards)
                for p in range(0, self.numberOfPlayers):
                    if p != observer:
                        numCards = len(CardHelper.to_list(hands[p]))
                        self.playerHands[p] = CardHelper.list_to_hand(listUnseenCards[:numCards])
                        listUnseenCards = listUnseenCards[numCards:]
                return True
            
            sums = self.getSumProbs(observer, probTables)
            listUnseenCards = CardHelper.to_list(unseenCards)
            random.shuffle(listUnseenCards)
            
            self.playerHands = [[] for i in range(0, self.numberOfPlayers)]
            self.playerHands[observer] = CardHelper.to_list(obsHand)
            dealerCards = []
            maxSum = max(sums)
            
            success = True
            while any([len(self.playerHands[i]) != lengths[i] for i in range(self.numberOfPlayers)]):
                if len(listUnseenCards) == 0:
                    listUnseenCards = dealerCards
                    dealerCards = []
                
                if len(listUnseenCards) == 0:
                    success = False
                    break
                
                card = listUnseenCards.pop()
                weights = [probTables[observer][i][card] + 0 for i in players]
                weights = [weights[i] / sums[i] * maxSum if sums[i] > 0 else 0 for i in range(len(players))]
                
                if sum(weights) < 1e-8:
                    success = False
                    break
                
                dealTo = random.choices(players, weights=weights, k=1)[0]
                
                if dealTo == self.numberOfPlayers:
                    dealerCards.append(card)
                else:
                    self.playerHands[dealTo].append(card)
                    if len(self.playerHands[dealTo]) == lengths[dealTo]:
                        players.remove(dealTo)
            
            if success and all([len(self.playerHands[i]) == lengths[i] for i in range(self.numberOfPlayers) if i != observer]):
                for i in self.players:
                    self.playerHands[i] = CardHelper.list_to_hand(self.playerHands[i])
                return True

    def CloneAndRandomize(self, observer):
        """Create a deep clone, randomizing information not visible to observer."""
        st = self.Clone()
        
        if st.useHeuristic[observer]:
            success = st.randomDeal(observer, self.probTables)
            if not success:
                # Fallback to uniform random
                seenCards = st.getSeenCards(observer)
                unseenCards = st.getUnseenCards(observer)
                listUnseenCards = CardHelper.to_list(unseenCards)
                random.shuffle(listUnseenCards)
                
                for p in range(0, st.numberOfPlayers):
                    if p != observer:
                        numCards = CardHelper.get_num_cards(self.playerHands[p])
                        st.playerHands[p] = CardHelper.list_to_hand(listUnseenCards[:numCards])
                        listUnseenCards = listUnseenCards[numCards:]
            return st
        else:
            seenCards = st.getSeenCards(observer)
            unseenCards = st.getUnseenCards(observer)
            listUnseenCards = CardHelper.to_list(unseenCards)
            random.shuffle(listUnseenCards)
            
            for p in range(0, st.numberOfPlayers):
                if p != observer:
                    numCards = CardHelper.get_num_cards(self.playerHands[p])
                    st.playerHands[p] = CardHelper.list_to_hand(listUnseenCards[:numCards])
                    listUnseenCards = listUnseenCards[numCards:]
            return st

    def GetCardDeck(self):
        """Construct a standard deck of 52 cards."""
        return sum(1 << i for i in range(0, 52))

    def getSideProb(self, numMySuits, numOppSuits):
        """Get conditional probability for side suits."""
        filtered = self.sideSuitProbs[
            (self.sideSuitProbs["numMySuit"] == numMySuits) & 
            (self.sideSuitProbs["numAtLeastOppSuit"] == numOppSuits)
        ]
        if filtered.empty:
            return 0
        else:
            return filtered["probability"].iloc[0]

    def getTrumpProb(self, numMySuits, numOppSuits):
        """Get conditional probability for trump suit."""
        filtered = self.trumpSuitProbs[
            (self.trumpSuitProbs["numMySuit"] == numMySuits) & 
            (self.trumpSuitProbs["numAtLeastOppSuit"] == numOppSuits)
        ]
        if filtered.empty:
            return 0
        else:
            return filtered["probability"].iloc[0]

    def Deal(self):
        """Reset the game state and deal cards."""
        self.playerToMove = self.GetNextPlayer(self.dealer)
        self.discards = 0
        self.currentTrick = []
        self.tricksTaken = [0 for p in range(0, self.numberOfPlayers)]
        self.bids = [-1 for p in range(0, self.numberOfPlayers)]  # Initialize to -1 (not yet bid)
        deck = CardHelper.to_list(self.GetCardDeck())
        deckMask = self.GetCardDeck()
        random.shuffle(deck)
        
        trumpCard = 0
        for i in range(0, self.numberOfPlayers):
            hand = []
            for j in range(self.tricksInRound):
                card = deck.pop()
                hand.append(card)
                deckMask = CardHelper.remove_card(deckMask, card)
            self.playerHands[i] = CardHelper.list_to_hand(hand)
        
        self.flippedOverCard = deck.pop()
        self.trumpSuit = CardHelper.get_card_suit(self.flippedOverCard, isHand=False)

    def getTrumpOneTrickProb(self,numPlayers,order,rank):
        """
        Gets the probability we will win the final trick given the number of players, the order in which we go, and the rank of our trump card
        """
        filtered = self.trumpOneTrickProbs[
            (self.trumpOneTrickProbs["num_players"] == numPlayers) & (
                        self.trumpOneTrickProbs["player_order"] == order ) & (self.trumpOneTrickProbs["rank"] == rank)]
        if filtered.empty:
            return 0
        else:
            return filtered["probability"].iloc[0]
    def getSideOneTrickProb(self,numPlayers,order,rank):
        """
                Gets the probability we will win the final trick given the number of players, the order in which we go, and the rank of our side suit card
                """
        filtered = self.sideOneTrickProbs[
            (self.sideOneTrickProbs["num_players"] == numPlayers) & (
                        self.sideOneTrickProbs["player_order"] == order ) & (self.trumpOneTrickProbs["rank"] == rank)]
        if filtered.empty:
            return 0
        else:
            return filtered["probability"].iloc[0]


    def Bid(self, bidsInPlayerOrder, p, startingPlayer):
        """
        Bidding algorithm used at the beginning of the game. p is the player order number
        (ex. if startingPlayer bids first, then p would be 0). startingPlayer is the player who is going to bid
        """
        expectedBid = 0
        myHand = self.playerHands[startingPlayer]
        if self.tricksInRound == 1:
            # If there is only one trick, we can fully determine what to do just using probability, so we treat it separately
            if p == self.numberOfPlayers - 1 and self.getBidSum() == 0:
                # If we are the last player to bid and nobody has bid, we cannot bid 1 as that would break the rules (can't have total bids sum to tricks)
                self.bids[startingPlayer] = 0
                bidsInPlayerOrder[p] = 0
            elif p == self.numberOfPlayers - 1 and self.getBidSum() == 1:
                # Same as above but now can't bid 0
                self.bids[startingPlayer] = 1
                bidsInPlayerOrder[p] = 1
            else:
                if self.getBidSum() < 1:
                    # If nobody has bid and we have a trump suit, effectively we can pretend like we are bidding first and the number of players
                    # is totalPlayers-peopleWho'veAlreadyBid. Look up probability and that is the bid
                    if CardHelper.get_card_suit(myHand) == self.trumpSuit:
                        self.bids[startingPlayer] = round(
                            self.getTrumpOneTrickProb(self.numberOfPlayers - p, 0, CardHelper.get_card_rank(myHand)))
                        bidsInPlayerOrder[p] = round(
                            self.getTrumpOneTrickProb(self.numberOfPlayers - p, 0, CardHelper.get_card_rank(myHand)))
                    # Otherwise, look up probability. This will always be less than 0.5, so really doesn't matter
                    else:
                        self.bids[startingPlayer] = round(
                            self.getSideOneTrickProb(self.numberOfPlayers, p, CardHelper.get_card_rank(myHand)))
                        bidsInPlayerOrder[p] = round(
                            self.getSideOneTrickProb(self.numberOfPlayers, p, CardHelper.get_card_rank(myHand)))
                else:
                    # If the first player didn't bid but someone else did, they must have a trump card
                    # (if acting rationally, which we assume). So we will also bid if we have a high trump
                    if self.getBidSum() == 1 and bidsInPlayerOrder[0] == 0:
                        if CardHelper.get_card_suit(myHand) == self.trumpSuit and CardHelper.get_card_rank(myHand) >= 8:
                            self.bids[startingPlayer] = 1
                            bidsInPlayerOrder[p] = 1
                    # If only the first player bid, we will also bid if we have a trump, though we might need
                    # a higher trump if there are more players in the game
                    elif self.getBidSum() == 1 and bidsInPlayerOrder[0] == 1:
                        if CardHelper.get_card_suit(myHand) == self.trumpSuit and self.numberOfPlayers == 2:
                            self.bids[startingPlayer] = 1
                            bidsInPlayerOrder[p] = 1
                        elif CardHelper.get_card_suit(myHand) == self.trumpSuit and self.numberOfPlayers == 3:
                            self.bids[startingPlayer] = 1
                            bidsInPlayerOrder[p] = 1
                        else:
                            if CardHelper.get_card_suit(myHand) == self.trumpSuit and CardHelper.get_card_rank(
                                    myHand) >= 8:
                                self.bids[startingPlayer] = 1
                                bidsInPlayerOrder[p] = 1
                    # If two (or more) players bid (very unlikely), then we better have a very high trump
                    elif self.getBidSum() == 2:
                        if bidsInPlayerOrder[0] == 0:
                            if CardHelper.get_card_suit(myHand) == self.trumpSuit and CardHelper.get_card_rank(
                                    myHand) >= 11:
                                self.bids[startingPlayer] = 1
                                bidsInPlayerOrder[p] = 1
                        else:
                            if CardHelper.get_card_suit(myHand) == self.trumpSuit and CardHelper.get_card_rank(
                                    myHand) >= 8:
                                self.bids[startingPlayer] = 1
                                bidsInPlayerOrder[p] = 1
                    elif self.getBidSum() >= 3:
                        if CardHelper.get_card_suit(myHand) == self.trumpSuit and CardHelper.get_card_rank(
                                myHand) >= 13:
                            self.bids[startingPlayer] = 1
                            bidsInPlayerOrder[p] = 1
        else:
            # More than 1 trick in round
            # Counting the number of cards we have in each suit, and checking to see which one is trump suit
            mySuits = {0: [0, False, 0],
                       1: [0, False, 0],
                       2: [0, False, 0],
                       3: [0, False, 0]}
            mySuits[self.trumpSuit][1] = True
            for suit, rank in CardHelper.iter_cards(myHand):
                mySuits[suit][0] += 1
                mySuits[suit][2] = CardHelper.add_card(mySuits[suit][2], CardHelper.to_card(suit, rank))
            for suit in [0, 1, 2, 3]:
                # Go through each suit and make bids with either two approaches: trump or off suit
                if suit != self.trumpSuit:
                    numCardsInSuit = mySuits[suit][0]
                    for suit, rank in CardHelper.iter_cards(mySuits[suit][2]):
                        # Go through each card in the off suit, and if it is a high card (>8) assign some expectedtricks it will win
                        # tempProb represents chance that all opponents will have 15-rankOfCard cards in that suit (this way we are guarenteed to win with that card)
                        # Specialprob models the chance that our card is higher rank than all other cards of that suit dealt, AND every player has at least 1 of that suit
                        # In the end, we take the higher of the two probabilities as our answer
                        tempProb = 0
                        specialProb = 1
                        if rank == 14:
                            tempProb = self.getSideProb(numCardsInSuit, 1)
                            specialProb = 0
                        elif rank == 13:
                            tempProb = self.getSideProb(numCardsInSuit, 2)
                            j = 0
                            k = 1
                            for i in range(14, 15):
                                if CardHelper.has_card(myHand, CardHelper.to_card(suit, i)):
                                    specialProb *= 1
                                    k += 1
                                else:
                                    specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                                52 - self.tricksInRound - j)
                                    j += 1
                                specialProb *= self.getSideProb(numCardsInSuit, k)
                        elif rank == 12:
                            tempProb = self.getSideProb(numCardsInSuit, 3)
                            j = 0
                            k = 1
                            for i in range(13, 15):
                                if CardHelper.has_card(myHand, CardHelper.to_card(suit, i)):
                                    specialProb *= 1
                                    k += 1
                                else:
                                    specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                            52 - self.tricksInRound - j)
                                    j += 1
                                specialProb *= self.getSideProb(numCardsInSuit, k)
                        elif rank == 11:
                            tempProb = self.getSideProb(numCardsInSuit, 4)
                            j = 0
                            k = 1
                            for i in range(12, 15):
                                if CardHelper.has_card(myHand, CardHelper.to_card(suit, i)):
                                    specialProb *= 1
                                    k += 1
                                else:
                                    specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                            52 - self.tricksInRound - j)
                                    j += 1
                                specialProb *= self.getSideProb(numCardsInSuit, k)
                        elif rank == 10:
                            tempProb = self.getSideProb(numCardsInSuit, 5)
                            j = 0
                            k = 1
                            for i in range(11, 15):
                                if CardHelper.has_card(myHand, CardHelper.to_card(suit, i)):
                                    specialProb *= 1
                                    k += 1
                                else:
                                    specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                            52 - self.tricksInRound - j)
                                    j += 1
                                specialProb *= self.getSideProb(numCardsInSuit, k)
                        elif rank == 9:
                            j = 0
                            k = 1
                            for i in range(10, 15):
                                if CardHelper.has_card(myHand, CardHelper.to_card(suit, i)):
                                    specialProb *= 1
                                    k += 1
                                else:
                                    specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                            52 - self.tricksInRound - j)
                                    j += 1
                                specialProb *= self.getSideProb(numCardsInSuit, k)
                        bv = 0
                        if rank >= 9:
                            if tempProb > specialProb:
                                # print(card,"Temp prob",tempProb)
                                bv = tempProb
                                expectedBid += tempProb
                            else:
                                # print(card, "special prob", specialProb)
                                bv = specialProb
                                expectedBid += specialProb
                        # print(CardHelper.to_str(CardHelper.to_card(suit,rank)),bv)
            # Here we deal with trump cards
            sideSuitCuts = {0: 1,
                            1: 1,
                            2: 1,
                            3: 1}
            trumpCardProbs = [[] for x in range(mySuits[self.trumpSuit][0])]
            count = 0
            # Two ways of winning a trump card: beating other trump cards (high card)
            # OR "trumping in" and winning as the only trump
            for trumpCard in sorted(CardHelper.to_list(mySuits[self.trumpSuit][2])):
                rank = trumpCard % 13 + 2
                suit = trumpCard // 13
                bestProb = 0
                bestProbSuit = ""
                # count each trump card with rank >= 10 as a win
                if rank >= 10:
                    trumpCardProbs[count].append(1)
                else:
                    for suit in [0, 1, 2, 3]:
                        if suit != self.trumpSuit:
                            k = mySuits[suit][0]
                            i = sideSuitCuts[suit]
                            prob = self.getSideProb(k, k + i)
                            if prob > bestProb:
                                bestProb = prob
                                bestProbSuit = suit
                            trumpCardProbs[count].append(prob)
                    if bestProb > 0:
                        sideSuitCuts[bestProbSuit] += 1
                count += 1
            trumpBids = 0
            for c in trumpCardProbs:
                # print(max(c))
                trumpBids += max(c)

            # having a minimum bid in situations where we have lots of middle to low trumps (algo thinks we should bid less than we should)
            minTrumpBid = 1.2 * mySuits[self.trumpSuit][0] ** 2 / self.tricksInRound
            if minTrumpBid > self.tricksInRound / 2:
                minTrumpBid = floor(self.tricksInRound / 2)
            if trumpBids < minTrumpBid:
                trumpBids = minTrumpBid
            # predBid is our predicted bid.
            # This is without considering other players bids or the position in which we bid
            predBid = trumpBids + expectedBid
            if p != 0:
                # Weighting by other players bids
                predBid -= (self.getBidSum() - (self.tricksInRound / self.numberOfPlayers * p)) / (
                            self.numberOfPlayers - p)
            if self.numberOfPlayers < 3:
                # empirically found weight to boost bid as without it, we consistently underbid
                predBid *= 1.3
            elif self.numberOfPlayers >= 3:
                predBid *= 1.13
            if self.bidStyle[startingPlayer] == "aggressive":
                predBid += 1
            elif self.bidStyle[startingPlayer] == "passive":
                predBid -= 1
            if predBid < 0:
                predBid = 0
            if predBid > self.tricksInRound:
                predBid = self.tricksInRound
            # Below we deal with the various scenarios that result when we are last player to bid
            # In Oh Hell, the total bids can't add up to the number of tricks, so the last player has
            # an additional constraint on their bidding
            if p == self.numberOfPlayers - 1:
                if round(predBid) + self.getBidSum() != self.tricksInRound:
                    if round(predBid) + self.getBidSum() - self.tricksInRound > 2:
                        predBid = self.tricksInRound + 2 - self.getBidSum()
                        self.bids[startingPlayer] = round(predBid)
                        bidsInPlayerOrder[p] = round(predBid)
                    elif round(predBid) + self.getBidSum() - self.tricksInRound < -2:
                        predBid = self.tricksInRound + 2 - self.getBidSum()
                        self.bids[startingPlayer] = round(predBid)
                        bidsInPlayerOrder[p] = round(predBid)
                    else:
                        self.bids[startingPlayer] = round(predBid)
                        bidsInPlayerOrder[p] = round(predBid)
                else:
                    if round(predBid) == 0:
                        self.bids[startingPlayer] = round(predBid) + 1
                        bidsInPlayerOrder[p] = round(predBid) + 1
                    elif self.getBidSum() == self.tricksInRound:
                        self.bids[startingPlayer] = round(predBid) + 1
                        bidsInPlayerOrder[p] = round(predBid) + 1
                    elif self.getBidSum() + predBid - self.tricksInRound >= 0:
                        self.bids[startingPlayer] = round(predBid) + 1
                        bidsInPlayerOrder[p] = round(predBid) + 1
                    else:
                        self.bids[startingPlayer] = round(predBid) - 1
                        bidsInPlayerOrder[p] = round(predBid) - 1
            else:
                self.bids[startingPlayer] = round(predBid)
                bidsInPlayerOrder[p] = round(predBid)

            # Sanity check to make sure that if we have a really high trump, we bid at least 1
            countHighTrump = 0
            for suit, rank in CardHelper.iter_cards(mySuits[self.trumpSuit][2]):
                if rank > 10:
                    countHighTrump += 1
            if countHighTrump > 0 and self.bids[startingPlayer] == 0:
                self.bids[startingPlayer] += 1
                bidsInPlayerOrder[p] = self.bids[startingPlayer]
                if self.getBidSum() == self.tricksInRound:
                    self.bids[startingPlayer] += 1
                    bidsInPlayerOrder[p] += 1

            # Another sanity check to make sure that we have at enough winners to bid this amount
            count = 0
            for suit, rank in CardHelper.iter_cards(myHand):
                if suit == self.trumpSuit:
                    count += 1
                if self.tricksInRound >= 8:
                    if rank > 11:
                        count += 1
                elif self.tricksInRound >= 6:
                    if rank > 10:
                        count += 1
                elif self.tricksInRound >= 4:
                    if rank > 9:
                        count += 1
                elif self.tricksInRound >= 2:
                    if rank > 8:
                        count += 1
                else:
                    count = 0
            if count < self.bids[startingPlayer]:
                diff = self.bids[startingPlayer] - count
                if self.getBidSum() - diff != self.tricksInRound:
                    self.bids[startingPlayer] = count
                else:
                    self.bids[startingPlayer] = count-1
                    if self.bids[startingPlayer] < 0:
                        if self.getBidSum() + 1 == self.tricksInRound:
                            self.bids[startingPlayer] = 1
                        else:
                            self.bids[startingPlayer] = 0
        if self.bids[startingPlayer] < 0:
            if self.getBidSum() + 1 == self.tricksInRound:
                self.bids[startingPlayer] = 1
            else:
                self.bids[startingPlayer] = 0
        if self.bids[startingPlayer] > self.tricksInRound:
            self.bids[startingPlayer] = self.tricksInRound
            if self.getBidSum() == self.tricksInRound:
                self.bids[startingPlayer] = self.tricksInRound-1
        self.haventBid[startingPlayer] = False

    def printHand(self,p):
        """
        Prints player p's hand in a readable way, grouping by suit and sorting within the suit
        """
        suit_order = ["C", "D", "H", "S"]  # or any order you prefer
        suit_names = {"S": "Spades", "H": "Hearts", "D": "Diamonds", "C": "Clubs"}
        # Group cards by suit
        cards_by_suit = {s: [] for s in suit_order}
        for suit,rank in CardHelper.iter_cards(self.playerHands[p]):
            cards_by_suit[suit_order[suit]].append(rank)

        # Sort each suit by rank (high to low)
        for suit_ind in range(len(suit_order)):
            s = suit_order[suit_ind]
            if self.trumpSuit == suit_ind:
                suit_names[s] = suit_names[s]+" -- Trump"
            cards_by_suit[s].sort(reverse=True)

        # Print result
        st = "Player " + str(p) + ": "
        for s in suit_order:
            cards_str = " ".join(f"{"??23456789TJQKA"[card]}{s}" for card in cards_by_suit[s])
            st += "[" + cards_str + "]" + " "
        # return st
        print(st)

        
    def getBidSum(self):
        return sum([bid for bid in self.bids if bid >=0])

    def updateProbTables(self, move, leadSuit):
        """Update probability tables after a card is played."""
        for observer in range(self.numberOfPlayers):
            if self.isAI[observer] and self.useHeuristic[observer]:
                self.onCardPlayed(move, observer)
                self.onVoid(move, observer, leadSuit)

    def onTrickWin(self, trickWinner):
        """Update probability tables after trick is won."""
        pass  # Simplified for server version

    def onCardPlayed(self, move, observer):
        """Update probabilities when a card is played."""
        for j in range(self.numberOfPlayers + 1):
            self.probTables[observer][j][move] = 0

    def onVoid(self, move, observer, leadSuit):
        """Update probabilities when a player shows void in a suit."""
        if CardHelper.get_card_suit(move, isHand=False) != leadSuit and not self.voids[leadSuit][self.playerToMove]:
            self.voids[leadSuit][self.playerToMove] = True
            for card in range(leadSuit * 13, (leadSuit + 1) * 13):
                if self.probTables[observer][self.playerToMove][card] > 1e-5:
                    change = -self.probTables[observer][self.playerToMove][card]
                    self.probTables[observer][self.playerToMove][card] += change
                    self.adjustProbabilities(observer, self.playerToMove, card, change)

    def DoMove(self, move):
        """Perform the given move and update game state."""
        self.currentTrick.append((self.playerToMove, move))
        self.playerHands[self.playerToMove] = CardHelper.remove_card(self.playerHands[self.playerToMove], move)
        
        if self.main and self.tricksInRound != 1 and self.tricksInRound - sum(self.tricksTaken) > 1:
            self.updateProbTables(move, CardHelper.get_card_suit(self.currentTrick[0][1], isHand=False))
        
        self.playerToMove = self.GetNextPlayer(self.playerToMove)
        
        # If we are at the end of a trick
        if any(True for (player, card) in self.currentTrick if player == self.playerToMove):
            winner = self.currentTrick[0][0]
            card1 = self.currentTrick[0][1]
            leadSuit = CardHelper.get_card_suit(card1, isHand=False)
            
            for (player, card2) in self.currentTrick:
                if not self.trickWinnerTable[card1][card2][leadSuit][self.trumpSuit]:
                    winner = player
                    card1 = card2 + 0
            
            trickWinner = winner
            if self.main and self.tricksInRound != 1:
                self.onTrickWin(trickWinner)
            
            self.tricksTaken[trickWinner] += 1
            
            for (player, card) in self.currentTrick:
                self.discards |= CardHelper.card_to_hand(card)
            
            self.currentTrick = []
            self.playerToMove = trickWinner

    def GetMoves(self):
        """Get all possible moves from this state."""
        hand = self.playerHands[self.playerToMove]
        
        if self.currentTrick == []:
            return hand
        else:
            (leader, leadCard) = self.currentTrick[0]
            cardsInSuit = CardHelper.get_cards_in_suit(CardHelper.get_card_suit(leadCard, isHand=False), hand)
            
            if cardsInSuit != 0:
                return cardsInSuit
            else:
                return hand

    def GetScore(self, player):
        """
        Get the game result from the viewpoint of player.
        Returns 1 if bid was made, 0 otherwise (for ISMCTS).
        """
        if self.tricksTaken[player] == self.bids[player]:
            return 1
        else:
            return 0

    def GetActualScore(self, player):
        """
        Get the actual Oh Hell score for the player.
        10 + tricks if bid was made, 0 otherwise.
        """
        if self.tricksTaken[player] == self.bids[player]:
            return 10 + self.tricksTaken[player]
        else:
            return 0

    def GetTricksNeeded(self, player):
        """Get how many more tricks the player needs."""
        return self.bids[player] - self.tricksTaken[player]
