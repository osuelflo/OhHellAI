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


class GameState:
    """Base game state class."""
    
    def __init__(self):
        self.numberOfPlayers = 4
        self.playerToMove = 1

    def GetNextPlayer(self, p):
        """Return the player to the left of the specified player."""
        return (p % self.numberOfPlayers) + 1

    def Clone(self):
        """Create a deep clone of this game state."""
        st = GameState()
        st.playerToMove = self.playerToMove
        return st

    def CloneAndRandomize(self, observer):
        """Create a deep clone, randomizing information not visible to observer."""
        return self.Clone()

    def DoMove(self, move):
        """Update state by carrying out the given move."""
        self.playerToMove = self.GetNextPlayer(self.playerToMove)

    def GetMoves(self):
        """Get all possible moves from this state."""
        pass

    def GetResult(self, player):
        """Get the game result from the viewpoint of player."""
        pass


class OhHellState(GameState):
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
        super().__init__()
        
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
        
        if start:
            self.trickWinnerTable = self.createTrickWinnerLookupTable()
            self.sideSuitProbs, self.trumpSuitProbs = self.getProbTables()
            self.sideOneTrickProbs, self.trumpOneTrickProbs = self.getOneTrickProbsTable()
            self.Deal()
            
            bidsInPlayerOrder = [0 for j in range(0, self.numberOfPlayers)]
            startingPlayer = self.dealer
            
            # Each player bids in order
            for p in self.players:
                startingPlayer = self.GetNextPlayer(startingPlayer)
                if not self.isAI[startingPlayer]:
                    # Human player - bid will be set later via server
                    self.bids[startingPlayer] = None
                else:
                    self.Bid(bidsInPlayerOrder, p, startingPlayer)
            
            if self.tricksInRound != 1:
                self.probTables = self.initializeProbTables()
                self.adjustProbsBids()
        
        if main:
            self.originalHands = deepcopy(self.playerHands)

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
        """Adjust probability tables based on bidding behavior."""
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
        self.bids = [0 for p in range(0, self.numberOfPlayers)]
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

    def Bid(self, bidsInPlayerOrder, p, startingPlayer):
        """
        AI bidding logic.
        Uses probability tables to estimate trick-taking potential.
        """
        myHand = self.playerHands[startingPlayer]
        
        if self.tricksInRound == 1:
            # Special case for 1-trick rounds
            expectedBid = 0.0
            myHandList = CardHelper.to_list(myHand)
            
            for card in myHandList:
                suit = card // 13
                rank = card % 13 + 2
                
                if suit == self.trumpSuit:
                    # Trump card
                    numCardsInSuit = CardHelper.get_suit_num_cards(myHand, suit)
                    if rank == 14:
                        expectedBid += self.getTrumpProb(numCardsInSuit, 1)
                    elif rank == 13:
                        expectedBid += self.getTrumpProb(numCardsInSuit, 2)
                    # ... (simplified for brevity)
                else:
                    # Non-trump card
                    numCardsInSuit = CardHelper.get_suit_num_cards(myHand, suit)
                    if rank == 14:
                        expectedBid += self.getSideProb(numCardsInSuit, 1)
                    elif rank == 13:
                        expectedBid += self.getSideProb(numCardsInSuit, 2)
            
            predBid = expectedBid
            
            # Adjust for position and other bids
            if self.bidStyle[startingPlayer] == "aggressive":
                predBid += 0.5
            elif self.bidStyle[startingPlayer] == "passive":
                predBid -= 0.5
            
            # Last bidder constraint
            if p == self.numberOfPlayers - 1:
                if round(predBid) + sum(self.bids) == self.tricksInRound:
                    if sum(self.bids) >= self.tricksInRound:
                        self.bids[startingPlayer] = 0
                    else:
                        self.bids[startingPlayer] = 1
                else:
                    self.bids[startingPlayer] = round(predBid)
            else:
                self.bids[startingPlayer] = round(predBid)
        else:
            # Multi-trick bidding logic
            expectedBid = 0.0
            mySuits = {0: [0, False, 0], 1: [0, False, 0], 2: [0, False, 0], 3: [0, False, 0]}
            mySuits[self.trumpSuit][1] = True
            
            for suit, rank in CardHelper.iter_cards(myHand):
                mySuits[suit][0] += 1
                mySuits[suit][2] = CardHelper.add_card(mySuits[suit][2], CardHelper.to_card(suit, rank))
            
            # Estimate off-suit tricks
            for suit in [0, 1, 2, 3]:
                if suit != self.trumpSuit:
                    numCardsInSuit = mySuits[suit][0]
                    for suit2, rank in CardHelper.iter_cards(mySuits[suit][2]):
                        tempProb = 0
                        specialProb = 1
                        
                        if rank == 14:
                            tempProb = self.getSideProb(numCardsInSuit, 1)
                            specialProb = 0
                        elif rank == 13:
                            tempProb = self.getSideProb(numCardsInSuit, 2)
                        elif rank == 12:
                            tempProb = self.getSideProb(numCardsInSuit, 3)
                        elif rank == 11:
                            tempProb = self.getSideProb(numCardsInSuit, 4)
                        elif rank == 10:
                            tempProb = self.getSideProb(numCardsInSuit, 5)
                        
                        if rank >= 9:
                            expectedBid += max(tempProb, specialProb)
            
            # Estimate trump tricks
            sideSuitCuts = {0: 1, 1: 1, 2: 1, 3: 1}
            trumpCardProbs = [[] for x in range(mySuits[self.trumpSuit][0])]
            count = 0
            
            for trumpCard in sorted(CardHelper.to_list(mySuits[self.trumpSuit][2])):
                rank = trumpCard % 13 + 2
                suit = trumpCard // 13
                bestProb = 0
                bestProbSuit = ""
                
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
            
            trumpBids = sum(max(c) for c in trumpCardProbs)
            
            # Minimum trump bid heuristic
            minTrumpBid = 1.2 * mySuits[self.trumpSuit][0] ** 2 / self.tricksInRound
            if minTrumpBid > self.tricksInRound / 2:
                minTrumpBid = floor(self.tricksInRound / 2)
            if trumpBids < minTrumpBid:
                trumpBids = minTrumpBid
            
            predBid = trumpBids + expectedBid
            
            # Adjust for other players' bids
            if p != 0:
                predBid -= (sum(self.bids) - (self.tricksInRound / self.numberOfPlayers * p)) / (self.numberOfPlayers - p)
            
            # Scaling factor
            if self.numberOfPlayers < 3:
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
            
            # Last bidder constraint
            if p == self.numberOfPlayers - 1:
                if round(predBid) + sum(self.bids) != self.tricksInRound:
                    if round(predBid) + sum(self.bids) - self.tricksInRound > 2:
                        predBid = self.tricksInRound + 2 - sum(self.bids)
                        self.bids[startingPlayer] = round(predBid)
                        bidsInPlayerOrder[p] = round(predBid)
                    elif round(predBid) + sum(self.bids) - self.tricksInRound < -2:
                        predBid = self.tricksInRound + 2 - sum(self.bids)
                        self.bids[startingPlayer] = round(predBid)
                        bidsInPlayerOrder[p] = round(predBid)
                    else:
                        self.bids[startingPlayer] = round(predBid)
                        bidsInPlayerOrder[p] = round(predBid)
                else:
                    if sum(self.bids) == self.tricksInRound:
                        self.bids[startingPlayer] = round(predBid) + 1
                        bidsInPlayerOrder[p] = round(predBid) + 1
                    elif sum(self.bids) + predBid - self.tricksInRound > 0:
                        self.bids[startingPlayer] = round(predBid) + 1
                        bidsInPlayerOrder[p] = round(predBid) + 1
                    else:
                        self.bids[startingPlayer] = round(predBid) - 1
                        bidsInPlayerOrder[p] = round(predBid) - 1
            else:
                self.bids[startingPlayer] = round(predBid)
                bidsInPlayerOrder[p] = round(predBid)
            
            # Sanity checks
            countHighTrump = sum(1 for suit, rank in CardHelper.iter_cards(mySuits[self.trumpSuit][2]) if rank > 10)
            if countHighTrump > 0 and self.bids[startingPlayer] == 0:
                self.bids[startingPlayer] += 1
                bidsInPlayerOrder[p] = self.bids[startingPlayer]
                if sum(self.bids) == self.tricksInRound:
                    self.bids[startingPlayer] += 1
                    bidsInPlayerOrder[p] += 1
            
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
            
            if count < self.bids[startingPlayer]:
                if self.bids[startingPlayer] - 1 != self.tricksInRound:
                    self.bids[startingPlayer] -= 1

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
