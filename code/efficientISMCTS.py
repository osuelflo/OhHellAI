# from heuristicISMCTS import *
# verbose = True
verboseHuman = True
verbose = False
# verboseHuman = False
from CardHelper import *
import numpy as np
import time
from colorama import Fore, Back, Style
from math import *
import random, sys
from copy import deepcopy
from Trickster import *

import numpy as np
import pandas as pd
# verbose = False
class Node:
    """
    A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
    """
    def __init__(self, move=None, parent=None, playerJustMoved=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.avails = 1
        self.playerJustMoved = playerJustMoved  # the only part of the state that the Node needs later
        self.triedMoves = []

    def GetUntriedMoves(self, legalMoves):
        """ Return the elements of legalMoves for which this node does not have children.
        """

        # Find all moves for which this node *does* have children
        triedMoves = self.triedMoves

        # Return all moves that are legal but have not been tried yet
        return [CardHelper.to_card(suit,rank) for suit,rank in CardHelper.iter_cards(legalMoves) if CardHelper.not_has_card(CardHelper.list_to_hand(triedMoves),CardHelper.to_card(suit,rank))]

    def UCBSelectChild(self, legalMoves, exploration=0.7):
        """ Use the UCB1 formula to select a child node, filtered by the given list of legal moves.
            exploration is a constant balancing between exploitation and exploration, with default value 0.7 (approximately sqrt(2) / 2)
        """

        # Filter the list of children by the list of legal moves
        legalChildren = [child for child in self.childNodes if CardHelper.has_card(legalMoves,child.move)]

        # Get the child with the highest UCB score
        s = max(legalChildren,
                key=lambda c: float(c.wins) / float(c.visits) + exploration * sqrt(log(c.avails) / float(c.visits)))

        # Update availability counts -- it is easier to do this now than during backpropagation
        for child in legalChildren:
            child.avails += 1

        # Return the child selected above
        return s

    def AddChild(self, m, p):
        """ Add a new child node for the move m.
            Return the added child node
        """
        n = Node(move=m, parent=self, playerJustMoved=p)
        self.childNodes.append(n)
        self.triedMoves.append(m)
        return n

    def Update(self, terminalState,score):
        """ Update this node - increment the visit count by one, and increase the win count by the result of terminalState for self.playerJustMoved.
        """
        self.visits += 1
        if self.playerJustMoved is not None:
            self.wins += score
            # self.wins += terminalState.GetScore(self.playerJustMoved)

    def __repr__(self):
        return "[M:%s W/V/A: %4i/%4i/%4i]" % (self.move, self.wins, self.visits, self.avails)

    def TreeToString(self, indent):
        """ Represent the tree as a string, for debugging purposes.
        """
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent + 1)
        return s

    def IndentString(self, indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s
def ISMCTS(rootstate, itermax, randomRollout,mainPlayer,verbose=False):
    """ Conduct an ISMCTS search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
    """

    rootnode = Node()

    for i in range(itermax):
        node = rootnode
        # Determinize
        state = rootstate.CloneAndRandomize(rootstate.playerToMove)
        # if node.untriedMoves == 0:
        #     node.untriedMoves = state.GetMoves()
        # Select

        while state.GetMoves() != 0 and node.GetUntriedMoves(
                state.GetMoves()) == []:  # node is fully expanded and non-terminal
            node = node.UCBSelectChild(state.GetMoves())
            state.DoMove(node.move)
        # Expand
        untriedMoves = node.GetUntriedMoves(state.GetMoves())
        if untriedMoves != []:  # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(untriedMoves)
            player = state.playerToMove
            state.DoMove(m)
            node = node.AddChild(m, player)  # add child and descend tree
        # Simulate
        if randomRollout:
            score = state.randomRollout(mainPlayer)
        else:
            while state.GetMoves() != 0:  # while state is non-terminal
                state.DoMove(random.choice(CardHelper.to_list(state.GetMoves())))
            score = state.GetScore(mainPlayer)
        # Backpropagate
        while node != None:  # backpropagate from the expanded node and work back to the root node
            node.Update(state,score)
            node = node.parentNode
    st = ""
    for node in rootnode.childNodes:
        m = node.move
        vis = node.visits
        wins = node.wins
        st += CardHelper.to_str(m)+ ": " +str(vis)+", "+ str(wins)+"| "
    # if verboseHuman:
    #     print(st)
    return max(rootnode.childNodes, key=lambda c: c.visits).move  # return the move that was most visited

class OhHellState():
    def __init__(self, n,numTricks,dealer,main,isAI,useHeuristic,bidStyle,enterCards,trickster,start = True):
        """ Initialise the game state.
        n is the number of players in the game.
        numTricks is the number of tricks to be played in the round (also the number of cards dealt to each player)
        numParticles is the number of particles used for the determinization phase in ISMCTS.
        lamb is the lambda parameter that controls how strongly the perceived strength of the opponent's hands is weighted in particle filtering
        dealer is the player who dealt
        main is a boolean that indicates whether this is the "main" state, or a cloned state created during ISMCTS.
        isAI is a boolean list that indicates whether each player is an AI or not
        useHeuristic is a boolean list that indicates whether each player should use particle filtering or not
        bidStyle is a list indicating whether each player bids normally, passively, or aggressively
        start is a boolean that indicates whether or not this is the start of the round and so we should deal out the cards and initialize particles
        """
        # Initializing all the class variables
        self.deck = self.GetCardDeck()
        self.enterCards = enterCards
        self.trickster = trickster
        self.bidStyle = bidStyle
        self.isAI = isAI
        self.main = main
        self.useHeuristic = useHeuristic
        self.players = list(range(0,n))
        self.dealer = dealer
        self.flippedOverCard = None
        self.numberOfPlayers = n
        # print(self.ESS, self.numberOfPlayers)
        self.playerToMove = 0
        self.tricksInRound = numTricks
        self.playerHands = [0 for p in range(0, self.numberOfPlayers)]
        self.discards = 0  # Stores the cards that have been played already in this round
        self.currentTrick = []
        self.trumpSuit = None
        self.tricksTaken = []  # Number of tricks taken by each player this round
        # self.knockedOut = {p: False for p in range(1, self.numberOfPlayers + 1)}

        self.bids = []
        self.voids = self.createVoids()
        if start:
            self.trickWinnerTable = self.createTrickWinnerLookupTable()
            self.sideSuitProbs, self.trumpSuitProbs = self.getProbTables()
            self.sideOneTrickProbs, self.trumpOneTrickProbs = self.getOneTrickProbsTable()
            self.Deal()
            if not all(self.isAI):
                print("Welcome to a game of Oh Hell!")
                print("\nThe flipped over card is", CardHelper.to_str(self.flippedOverCard))
            bidsInPlayerOrder = [0 for j in range(0, self.numberOfPlayers)]
            startingPlayer = self.dealer
            # Each player bids in order, asking human player for bid if p is human
            for p in self.players:
                startingPlayer = self.GetNextPlayer(startingPlayer)
                if not self.isAI[startingPlayer]:
                    print("Here is your hand:")
                    self.printHand(startingPlayer)
                    bid = self.askForBid()
                    self.bids[startingPlayer] = bid
                    bidsInPlayerOrder[p] = bid
                else:
                    if self.trickster[startingPlayer]:
                        bid = self.makeTricksterBid(startingPlayer)
                        self.bids[startingPlayer] = bid
                        bidsInPlayerOrder[startingPlayer] = bid
                    else:
                        self.Bid(bidsInPlayerOrder,p,startingPlayer)
                if self.bids[startingPlayer] < 0:
                    print("FUCK")
                    print(startingPlayer)
                    print(self.trickster[startingPlayer])
            assert sum(self.bids) != self.tricksInRound
            # If we only have one card, that is the only move we can make so must play it.
            if self.tricksInRound != 1:
                self.probTables = self.initializeProbTables()
                self.adjustProbsBids()
        if main:
            self.originalHands = deepcopy(self.playerHands)
            # self.printProbTables(self.probTables,0,3)

    def makeTricksterBid(self,player):
        tricksterBot = CSharpBot(self,player)
        bid = tricksterBot.suggest_bid()
        return bid
    def compressFactor(self,factor):
        import math
        x_min, x_max = 0.05, 15
        y_min, y_max = 0.5, 2
        log_x_min = math.log(x_min)
        log_x_max = math.log(x_max)
        a = (math.log(y_max) - math.log(y_min)) / (log_x_max - log_x_min)
        b = math.log(y_min) - a * log_x_min
        return math.exp(a * math.log(factor) + b)

    def adjustProbabilities(self,observer,player,card,change,probTables = None):
        """
        Plan:
        1. p_i = p_i(1-change/sum(p_i))
        """
        if probTables == None:
            sum_pi = sum([self.probTables[observer][i][card] for i in range(self.numberOfPlayers+1) if i != player])
            if sum_pi > 1e-8:
                for i in range(self.numberOfPlayers+1):
                    if i != player and i != observer:
                        self.probTables[observer][i][card] += -change*self.probTables[observer][i][card]/sum_pi
            # assert abs(sum([self.probTables[observer][i][card] for i in range(self.numberOfPlayers+1)])-1) < 1e-4
        else:
            sum_pi = sum([probTables[observer][i][card] for i in range(self.numberOfPlayers + 1) if i != player])
            if sum_pi > 1e-8:
                for i in range(self.numberOfPlayers + 1):
                    if i != player and i != observer:
                        probTables[observer][i][card] += -change*probTables[observer][i][card]/ sum_pi
                assert abs(
                    sum([probTables[observer][i][card] for i in range(self.numberOfPlayers + 1)]) - 1) < 1e-4
            return probTables
    def getSumProbs(self,observer,probTables):
        return [sum(probTables[observer][i]) for i in range(self.numberOfPlayers+1)]
    def probChange(self,factor,prob):
        return 1/(1+exp(-(log(prob/(1-prob))+factor))) - prob
    def adjustProbsBids(self):
        """
        Plan:
        1. Things that impact the perceived strength of opponent hand
            - number of tricks bid
            - position in which they bid
            - number of cards dealt
            - my own bid
            - number of prior bids
        2. If bid is above average (opposite for below average)
            - trump cards UP
            - high cards UP
            - if observer has a lot of a specific suit, maybe that suit for the high bid player has less probability (short suited = higher bid)
        """
        p = self.GetNextPlayer(self.dealer)
        playOrder = [p]
        while p != self.dealer:
            p = self.GetNextPlayer(p)
            playOrder.append(p)
        for observer in range(self.numberOfPlayers):
            if self.isAI[observer] and self.useHeuristic[observer]:
                # self.printHand(observer)
                for p in range(self.numberOfPlayers):
                    player = playOrder[p]
                    if player != observer:
                        bid = self.bids[playOrder[p]]
                        expBid = (self.tricksInRound-sum(self.bids[:(p)]))/(self.numberOfPlayers-p)
                        if expBid <= 0:
                            expBid = 1/(self.tricksInRound/self.numberOfPlayers)*2
                        expBidFactor = (bid / expBid)
                        for card in range(52):
                            # print(card)
                            if not CardHelper.has_card(self.playerHands[observer],card) and card != self.flippedOverCard:
                                rank = CardHelper.get_card_rank(card,isHand=False)
                                suit = CardHelper.get_card_suit(card,isHand=False)
                                trumpFactor = 1
                                if suit == self.trumpSuit:
                                    trumpFactor = 1.5
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
                                observerSuits = CardHelper.get_suit_num_cards(self.playerHands[observer],suit)
                                suitFactor = 1
                                if observerSuits > 1.5*self.tricksInRound/4 and expBidFactor > 1:
                                    suitFactor = 1/(expBidFactor*(observerSuits/(1.5*self.tricksInRound/4)))*1.2
                                if expBidFactor == 0:
                                    expBidFactor = 0.75/expBid
                                if expBidFactor > 1 and rank >= 8:
                                    totalFactor = (expBidFactor*rankFactor*suitFactor*trumpFactor)
                                elif expBidFactor > 1 and rank < 8:
                                    totalFactor = (rankFactor*suitFactor/expBidFactor*trumpFactor)
                                elif expBidFactor < 1 and rank < 8:
                                    totalFactor = suitFactor/expBidFactor/rankFactor/trumpFactor
                                elif expBidFactor < 1 and rank >= 8:
                                    totalFactor = (expBidFactor/rankFactor*suitFactor/trumpFactor)
                                else:
                                    totalFactor = 1
                                if totalFactor <= 0:
                                    print(totalFactor,expBid,expBidFactor,suit,rank,suitFactor,trumpFactor,rankFactor)
                                    print(self.tricksInRound,self.numberOfPlayers)
                                    self.printHand(p)
                                    self.printHand(observer)
                                totalFactor = self.compressFactor(totalFactor)
                                change = self.probChange(totalFactor,self.probTables[observer][player][card])
                                self.probTables[observer][player][card]+=change
                                self.adjustProbabilities(observer,player,card,change)
    def initializeProbTables(self):
        """
        Plan:
        1. For each AI player, have a (num_players+1) x 52 card array containing probabilities that they have that card
        2. One of the players will be the cards that aren't dealt
        3. Initialize the cards properly
        """
        probCardDealt = (self.numberOfPlayers-1)*self.tricksInRound/(51-self.tricksInRound)
        probCardDealtPlayer = probCardDealt/(self.numberOfPlayers-1)
        probCardNotDealt = 1-probCardDealt
        probTables = []
        for player in range(self.numberOfPlayers):
            playerHand = self.playerHands[player]
            if self.isAI[player]:
                probTables.append([[0 for i in range(52)] for j in range(self.numberOfPlayers+1)])
                for p in range(self.numberOfPlayers+1):
                    for card in range(52):
                        if card == self.flippedOverCard:
                            if p == self.numberOfPlayers:
                                probTables[player][p][card] = 0
                            else:
                                probTables[player][p][card] = 0
                        elif p == player:
                            probTables[player][p][card] = int(CardHelper.has_card(playerHand,card))
                        else:
                            if CardHelper.has_card(playerHand,card):
                                probTables[player][p][card] = 0
                            else:
                                if p == self.numberOfPlayers:
                                    probTables[player][p][card] = probCardNotDealt
                                else:
                                    probTables[player][p][card] = probCardDealtPlayer
            else:
                probTables.append([[0 for i in range(52)] for j in range(self.numberOfPlayers + 1)])
        return probTables
    def randomRollout(self,player):
        """
        Plan for rollout:
        1. Check if `player` can even win its desired tricks. If not, return their score (0)
        2. Then check if the available moves only contains one move. If so, play that move
        3. If `player` needs to win tricks,
            - play the highest rank card in their available moves, IF doing so would win the trick as is. IF they would lose
            no matter what, play the lowest rank card.
        4. If `player` needs to not win any more tricks,
            - play highest rank card in their available moves, such that they will still lose.
        """
        while self.GetMoves() != 0:  # while state is non-terminal
            tricksLeftPlayer = CardHelper.get_num_cards(self.playerHands[self.playerToMove])
            tricksNeededPlayer = self.bids[player]-self.tricksTaken[player]
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
                        trumpCards = CardHelper.get_cards_in_suit(self.trumpSuit,moves)
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
    def checkWinTrick(self,card2):
        winner = self.currentTrick[0][0]
        card = self.currentTrick[0][1]
        leadSuit = CardHelper.get_card_suit(card, isHand=False)
        for (player, card1) in self.currentTrick:
            # card1 = CardHelper.hand_to_card(card)
            if self.trickWinnerTable[card1][card2][leadSuit][self.trumpSuit]:
                return False
        return True
    def getWinnerMidTrick(self):
        card1 = self.currentTrick[0][1]
        winner = self.currentTrick[0][0]
        leadSuit = CardHelper.get_card_suit(card1, isHand=False)
        for (player, card2) in self.currentTrick:
            # card2 = CardHelper.hand_to_card(card)
            if not self.trickWinnerTable[card1][card2][leadSuit][self.trumpSuit]:
                winner = player
                card1 = card2
        return card1,winner
    def createTrickWinnerLookupTable(self):
        WINNERS = np.zeros((52, 52, 4, 4))
        for c1 in range(52):
            for c2 in range(52):
                for lead in range(4):
                    for trump in range(4):
                        WINNERS[c1][c2][lead][trump] = CardHelper.card_wins(c1, c2, lead, trump)
        return WINNERS
    def askForBid(self):
        """
        Asks the user for a bid
        """
        bid = int(input("What would you like to Bid? "))
        return bid
    def createVoids(self):
        """
        Initializes the boolean vector for whether or not a certain player is void in a certain suit
        """
        voids = {0: [False for p in range(0, self.numberOfPlayers)],
                 1: [False for p in range(0, self.numberOfPlayers)],
                 2: [False for p in range(0, self.numberOfPlayers)],
                 3: [False for p in range(0, self.numberOfPlayers)]}
        return voids
    def isTrump(self,suit):
        """Checks whether suit is the trump suit"""
        if suit == self.trumpSuit:
            return "T"
        else:
            return "O"
    def getSeenCards(self,observer):
        """
        Returns all cards that the observer has seen. Its own, the cards in the current trick,
        the cards that have been already played in previous tricks, and the flipped over card
        """
        return self.discards | self.playerHands[observer] | CardHelper.list_to_hand(c for (player, c) in self.currentTrick) | CardHelper.card_to_hand(self.flippedOverCard)
        # return [card for card in self.GetCardDeck() if
        #                card in self.discards or card in self.playerHands[
        #                    observer] or card in [c for (player, c) in self.currentTrick] or card == self.flippedOverCard]
    def getUnseenCards(self,observer):
        """
        Gets all cards that the observer has not seen. Note that not all cards are actually dealt.
        """

        return CardHelper.get_difference(self.deck,self.getSeenCards(observer))
        return self.deck & ~self.discards & ~self.playerHands[observer] & ~sum(
            c for (player, c) in self.currentTrick) & ~self.flippedOverCard
        # return [card for card in self.GetCardDeck() if
        #         card not in self.discards and card not in self.playerHands[
        #             observer] and card not in [c for (player, c) in self.currentTrick] and card != self.flippedOverCard]
    def getOneTrickProbsTable(self):
        """
        Returns a probability table to help in the bidding process for tricks with only one card
        """
        return pd.read_csv("probabilityData/sideOneTrickProbs.csv"),pd.read_csv("probabilityData/trumpOneTrickProbs.csv")
    def getProbTables(self):
        """
        Gets the side suit and trump suit probability tables for the current combination of
        trick in round and number of players. Each table contains probabilities of all the rest of the players holding
        X number of cards of suit (trump or off) given we have Y # cards in that suit.
        Useful information for bidding.
        """
        try:
            df = pd.read_csv(f"probabilityData/{self.tricksInRound}Tricks{self.numberOfPlayers}PSideSuit.csv")
            df2 = pd.read_csv(f"probabilityData/{self.tricksInRound}Tricks{self.numberOfPlayers}PTrumpSuit.csv")
        except pd.errors.EmptyDataError:
            print("FUCK")
            print(self.tricksInRound,self.numberOfPlayers)
        return df,df2
    def Clone(self):
        """
        Create a deep clone of this game state.
        Notably does not clone anything related to particle filtering as this information is not needed for
        game states solely used for ISMCTS simulation.
        """

        st = OhHellState(self.numberOfPlayers,self.tricksInRound,self.dealer,False,self.isAI,self.useHeuristic,self.bidStyle,self.enterCards,self.trickster,False)
        st.trickWinnerTable = self.trickWinnerTable
        st.deck = self.deck
        st.enterCards = self.enterCards
        st.trickster = self.trickster
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

    def reMakeProbTable(self,observer,opponent,probTables):
        print("remaking")
        probTablesT = deepcopy(probTables)
        unseen = self.getUnseenCards(observer)
        seen = self.getSeenCards(observer)
        baseProb = CardHelper.get_num_cards(self.playerHands[observer])/CardHelper.get_num_cards(self.getUnseenCards(observer))
        for card in range(52):
            if CardHelper.has_card(unseen,card) and not self.voids[CardHelper.get_card_suit(card,isHand=False)][opponent]:
                change = -probTablesT[observer][opponent][card] +baseProb
                probTablesT[observer][opponent][card] += change
                probTablesT = self.adjustProbabilities(observer,opponent,card,change,probTables=probTablesT)
        # print(probTables[observer][opponent])
        assert all([probTables[observer][opponent][i] < 1 for i in range(52)])
        bid = self.GetTricksNeeded(opponent)
        expBid = (self.tricksInRound - sum(self.tricksTaken)) / (self.numberOfPlayers)
        # if expBid <= 0:
        #     expBid = 1 / (self.tricksInRound / self.numberOfPlayers) * 2
        expBidFactor = (bid / expBid)
        for card in range(52):
            if CardHelper.has_card(unseen,card) and not self.voids[CardHelper.get_card_suit(card,isHand=False)][opponent]:
                rank = CardHelper.get_card_rank(card, isHand=False)
                suit = CardHelper.get_card_suit(card, isHand=False)
                print(rank,CardHelper.get_str_suit(suit))
                trumpFactor = 1
                if suit == self.trumpSuit:
                    trumpFactor = 1.5
                rankFactor = rank / 8
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
                totalFactor /= 4
                if totalFactor <= 0:
                    totalFactor = 1
                totalFactor = self.compressFactor(totalFactor)
                change = self.probChange(totalFactor, probTablesT[observer][opponent][card])
                probTablesT[observer][opponent][card] += change
                print(probTablesT[observer][opponent][card])
                probTablesT = self.adjustProbabilities(observer, opponent, card, change,probTables = probTablesT)
            else:
                probTablesT[observer][opponent][card] = 0
        self.printProbTables(probTablesT,observer,opponent)
        return probTablesT

    def randomDeal(self,observer,probTables):
        unseenCards = self.getUnseenCards(observer)
        lengths = [CardHelper.get_num_cards(self.playerHands[i]) for i in range(self.numberOfPlayers)]
        players = self.players + [self.numberOfPlayers]
        players.remove(observer)
        badcount = 0
        obsHand = self.playerHands[observer]
        hands = deepcopy(self.playerHands)
        bad = True
        count = 0
        poop = False
        startTwo = time.perf_counter()
        while bad:
            if startTwo - time.perf_counter() > 2:
                return False
            start = time.perf_counter()
            sums = self.getSumProbs(observer, probTables)

            listUnseenCards = CardHelper.to_list(unseenCards)
            random.shuffle(listUnseenCards)
            # plan: pick the first card from the deck of shuffled cards
            # give it to one of the players (or the undealt cards)
            # make sure to stop giving cards to players who have enough cards already
            # if all cards have been dealt but player(s) still need cards, redeal using the cards dealt to the "dealer"
            if poop:
                print("yeet")
                print(sums)
                print(badcount)
                if badcount > 100:
                    for p in range(0, self.numberOfPlayers):
                        if p != observer:
                            # Deal cards to player p
                            # Store the size of player p's hand
                            numCards = len(CardHelper.to_list(hands[p]))
                            # Give player p the first numCards unseen cards
                            self.playerHands[p] = CardHelper.list_to_hand(listUnseenCards[: numCards])
                            # Remove those cards from unseenCards
                            listUnseenCards = listUnseenCards[numCards:]
                    return
            poop = False
            self.playerHands = [[] for i in range(0, self.numberOfPlayers)]
            self.playerHands[observer] = CardHelper.to_list(obsHand)
            dealerCards = []
            maxSum = max(sums)

            # Check if any probtables have less than lengths[i] cards with non zero probs. If so, remake them.
            while any([len(self.playerHands[i]) != lengths[i] for i in range(self.numberOfPlayers)]):
                # print(self.playerHands)
                # print("\n")
                if len(listUnseenCards) == 0:
                    listUnseenCards = dealerCards
                    dealerCards = []
                if len(listUnseenCards) == 0:
                    bad = True
                    players = self.players + [self.numberOfPlayers]
                    players.remove(observer)
                    break
                card = listUnseenCards.pop()
                shat = False
                for i in range(len(players)):
                    if sums[i] == 0:
                        self.playerHands[observer] = CardHelper.list_to_hand(self.playerHands[observer])

                        print("RE TRY?>>>>>>>>>>>>>>>>>")
                        self.printProbTables(probTables, observer, i)
                        newTables = self.reMakeProbTable(observer, i, probTables)
                        probTables[:] = newTables
                        bad = True
                        players = self.players + [self.numberOfPlayers]
                        players.remove(observer)
                        shat = True
                        break
                if shat:
                    break
                weights = [probTables[observer][i][card]+0 for i in players]
                weights = [weights[i]/sums[i]*maxSum for i in range(len(players))]
                # if sum(weights)< 1e-8:
                #     for pl in players:
                #         self.printProbTables(probTables,observer,pl)
                dealTo = random.choices(players,weights = weights,k=1)[0]
                if dealTo == self.numberOfPlayers:
                    dealerCards.append(card)
                else:
                    self.playerHands[dealTo].append(card)
                    if len(self.playerHands[dealTo]) == lengths[dealTo]:
                        players.remove(dealTo)
                if time.perf_counter() - start > 0.001:
                    count += 1
                    if count > 100:
                        poop = True
                        count = 0
                        print("foooook")
                        badcount += 1
                        self.playerHands[observer] = CardHelper.list_to_hand(self.playerHands[observer])
                        print(dealTo)
                        # breakpoint()
                        c = 0
                        fix = False

                        for i in players:
                            if i != self.numberOfPlayers:
                                for cd in range(52):
                                    if CardHelper.has_card(unseenCards,cd):
                                        if probTables[observer][i][cd] > 0:
                                            c += 1
                                if c < lengths[i]:
                                    fix = True
                                    print(lengths[i])
                                    print(c)
                                    print(self.playerHands[i])
                                    print(i)
                                    self.printProbTables(probTables,observer,i)
                                    newTables = self.reMakeProbTable(observer, i, probTables)
                                    probTables[:] = newTables
                                c = 0
                        if not fix:
                            options = []
                            for i in players:
                                if i != self.numberOfPlayers:
                                    for cd in range(52):
                                        if CardHelper.has_card(unseenCards, cd):
                                            if probTables[observer][i][cd] > 0:
                                                c += 1
                                    if c >= 1:
                                        options.append(i)
                                    c = 0
                            pl = random.choices(options)[0]
                            newTables = self.reMakeProbTable(observer, pl, probTables)
                            probTables[:] = newTables
                    bad = True
                    players = self.players + [self.numberOfPlayers]
                    players.remove(observer)
                    break
            if all([len(self.playerHands[i]) == lengths[i] for i in range(self.numberOfPlayers) if i != observer]):
                bad = False
        for i in self.players:
            self.playerHands[i] = CardHelper.list_to_hand(self.playerHands[i])
        return True
        # if count > 1:
        #     print(count)
    def CloneAndRandomize(self, observer):
        """ Create a deep clone of this game state, randomizing any information not visible to the specified observer player if they aren't using heuristic.
        If they are, then sample from the particles based on their weights.
        """
        st = self.Clone()
        if st.useHeuristic[observer]:
            # sample from particles
            # seenCards = st.getSeenCards(observer)
            # seenCards = st.playerHands[observer] + st.discards + [st.flippedOverCard] + [card for (player, card) in
            #                                                                              st.currentTrick]
            # particle = deepcopy(random.choices(self.particles[observer], self.weights[observer], k =1))[0]
            # for p in range(0,st.numberOfPlayers):
            #     st.playerHands[p] = particle[p]
            success = st.randomDeal(observer,self.probTables)
            if not success:
                seenCards = st.getSeenCards(observer)
                # The observer can't see the rest of the deck
                unseenCards = st.getUnseenCards(observer)
                listUnseenCards = CardHelper.to_list(unseenCards)
                # Deal the unseen cards to the other players
                random.shuffle(listUnseenCards)
                for p in range(0, st.numberOfPlayers):
                    if p != observer:
                        # Deal cards to player p
                        # Store the size of player p's hand
                        numCards = CardHelper.get_num_cards(self.playerHands[p])
                        # Give player p the first numCards unseen cards
                        st.playerHands[p] = CardHelper.list_to_hand(listUnseenCards[: numCards])
                        # Remove those cards from unseenCards
                        unseenCards = listUnseenCards[numCards:]
            return st
        else:
            # The observer can see his own hand and the cards in the current trick, and can remember the cards played in previous tricks
            # seenCards = st.playerHands[observer] + st.discards + [st.flippedOverCard] + [card for (player, card) in
            #                                                                              st.currentTrick]
            seenCards = st.getSeenCards(observer)
            # The observer can't see the rest of the deck
            unseenCards = st.getUnseenCards(observer)
            listUnseenCards = CardHelper.to_list(unseenCards)
            # Deal the unseen cards to the other players
            random.shuffle(listUnseenCards)
            for p in range(0, st.numberOfPlayers):
                if p != observer:
                    # Deal cards to player p
                    # Store the size of player p's hand
                    numCards = CardHelper.get_num_cards(self.playerHands[p])
                    # Give player p the first numCards unseen cards
                    st.playerHands[p] = CardHelper.list_to_hand(listUnseenCards[: numCards])
                    # Remove those cards from unseenCards
                    unseenCards = listUnseenCards[numCards:]
            return st
    def GetCardDeck(self):
        """ Construct a standard deck of 52 cards.
        """
        return sum(1 << i for i in range(0, 52))
        # return [Card(rank, suit) for rank in range(2, 14 + 1) for suit in ['C', 'D', 'H', 'S']]
    def getSideProb(self,numMySuits,numOppSuits):
        """
        Gets the conditional probability P(min(#oppCardsInSuit)|#myCardsInSuit)
        if we are looking at a side suit
        """
        filtered = self.sideSuitProbs[(self.sideSuitProbs["numMySuit"]==numMySuits) & (self.sideSuitProbs["numAtLeastOppSuit"]==numOppSuits)]
        if filtered.empty:
            return 0
        else:
            return filtered["probability"].iloc[0]
    def getTrumpProb(self,numMySuits,numOppSuits):
        """
                Gets the conditional probability P(min(#oppCardsInSuit)|#myCardsInSuit)
                if we are looking at the trump suit
        """
        filtered = self.trumpSuitProbs[
            (self.trumpSuitProbs["numMySuit"] == numMySuits) & (self.trumpSuitProbs["numAtLeastOppSuit"] == numOppSuits)]
        if filtered.empty:
            return 0
        else:
            return filtered["probability"].iloc[0]
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

    def Bid(self,bidsInPlayerOrder, p,startingPlayer):
        """
        Bidding algorithm used at the beginning of the game. p is the player order number
        (ex. if startingPlayer bids first, then p would be 0). startingPlayer is the player who is going to bid
        """
        expectedBid = 0
        myHand = self.playerHands[startingPlayer]
        if verbose:
            self.printHand(startingPlayer)
        if self.tricksInRound == 1:
            # If there is only one trick, we can fully determine what to do just using probability, so we treat it separately
            if p == self.numberOfPlayers - 1 and sum(self.bids) == 0:
                # If we are the last player to bid and nobody has bid, we cannot bid 1 as that would break the rules (can't have total bids sum to tricks)
                self.bids[startingPlayer] = 0
                bidsInPlayerOrder[p] = 0
            elif p == self.numberOfPlayers - 1 and sum(self.bids) ==1:
                # Same as above but now can't bid 0
                self.bids[startingPlayer] = 1
                bidsInPlayerOrder[p] = 1
            else:
                if sum(self.bids) < 1:
                    # If nobody has bid and we have a trump suit, effectively we can pretend like we are bidding first and the number of players
                    # is totalPlayers-peopleWho'veAlreadyBid. Look up probability and that is the bid
                    if CardHelper.get_card_suit(myHand) == self.trumpSuit:
                        self.bids[startingPlayer] = round(self.getTrumpOneTrickProb(self.numberOfPlayers-p,0,CardHelper.get_card_rank(myHand)))
                        bidsInPlayerOrder[p] = round(self.getTrumpOneTrickProb(self.numberOfPlayers-p,0,CardHelper.get_card_rank(myHand)))
                    # Otherwise, look up probability. This will always be less than 0.5, so really doesn't matter
                    else:
                        self.bids[startingPlayer] = round(self.getSideOneTrickProb(self.numberOfPlayers, p, CardHelper.get_card_rank(myHand)))
                        bidsInPlayerOrder[p] = round(
                            self.getSideOneTrickProb(self.numberOfPlayers, p, CardHelper.get_card_rank(myHand)))
                else:
                    # If the first player didn't bid but someone else did, they must have a trump card
                    # (if acting rationally, which we assume). So we will also bid if we have a high trump
                    if sum(self.bids) == 1 and bidsInPlayerOrder[0] == 0:
                        if CardHelper.get_card_suit(myHand) == self.trumpSuit and CardHelper.get_card_rank(myHand) >= 8:
                            self.bids[startingPlayer] = 1
                            bidsInPlayerOrder[p] = 1
                    # If only the first player bid, we will also bid if we have a trump, though we might need
                    # a higher trump if there are more players in the game
                    elif sum(self.bids) == 1 and bidsInPlayerOrder[0] == 1:
                        if CardHelper.get_card_suit(myHand) == self.trumpSuit and self.numberOfPlayers == 2:
                            self.bids[startingPlayer] = 1
                            bidsInPlayerOrder[p] = 1
                        elif CardHelper.get_card_suit(myHand) == self.trumpSuit and self.numberOfPlayers == 3:
                            self.bids[startingPlayer] = 1
                            bidsInPlayerOrder[p] = 1
                        else:
                            if CardHelper.get_card_suit(myHand) == self.trumpSuit and CardHelper.get_card_rank(myHand) >= 8:
                                self.bids[startingPlayer] = 1
                                bidsInPlayerOrder[p] = 1
                    # If two (or more) players bid (very unlikely), then we better have a very high trump
                    elif sum(self.bids) == 2:
                        if bidsInPlayerOrder[0] == 0:
                            if CardHelper.get_card_suit(myHand) == self.trumpSuit and CardHelper.get_card_rank(myHand) >= 11:
                                self.bids[startingPlayer] = 1
                                bidsInPlayerOrder[p] = 1
                        else:
                            if CardHelper.get_card_suit(myHand) == self.trumpSuit and CardHelper.get_card_rank(myHand) >= 8:
                                self.bids[startingPlayer] = 1
                                bidsInPlayerOrder[p] = 1
                    elif sum(self.bids) >= 3:
                        if CardHelper.get_card_suit(myHand) == self.trumpSuit and CardHelper.get_card_rank(myHand) >= 13:
                            self.bids[startingPlayer] = 1
                            bidsInPlayerOrder[p] = 1
        else:
            # More than 1 trick in round
            # Counting the number of cards we have in each suit, and checking to see which one is trump suit
            mySuits = {0:[0,False,0],
                       1:[0,False,0],
                       2:[0,False,0],
                       3:[0,False,0]}
            mySuits[self.trumpSuit][1] = True
            for suit, rank in CardHelper.iter_cards(myHand):
                mySuits[suit][0] += 1
                mySuits[suit][2] = CardHelper.add_card(mySuits[suit][2], CardHelper.to_card(suit, rank))
            for suit in [0,1,2,3]:
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
                            k=1
                            for i in range(14,15):
                                if CardHelper.has_card(myHand,CardHelper.to_card(suit, i)):
                                    specialProb *= 1
                                    k+=1
                                else:
                                    specialProb *= 1 - (self.tricksInRound*(self.numberOfPlayers-1))/(52-self.tricksInRound-j)
                                    j += 1
                                specialProb *= self.getSideProb(numCardsInSuit, k)
                        elif rank == 12:
                            tempProb = self.getSideProb(numCardsInSuit, 3)
                            j = 0
                            k=1
                            for i in range(13, 15):
                                if CardHelper.has_card(myHand,CardHelper.to_card(suit, i)):
                                    specialProb *= 1
                                    k+=1
                                else:
                                    specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                                52 - self.tricksInRound - j)
                                    j += 1
                                specialProb *= self.getSideProb(numCardsInSuit, k)
                        elif rank == 11:
                            tempProb = self.getSideProb(numCardsInSuit, 4)
                            j = 0
                            k=1
                            for i in range(12, 15):
                                if CardHelper.has_card(myHand,CardHelper.to_card(suit, i)):
                                    specialProb *= 1
                                    k+=1
                                else:
                                    specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                                52 - self.tricksInRound - j)
                                    j += 1
                                specialProb *= self.getSideProb(numCardsInSuit, k)
                        elif rank == 10:
                            tempProb = self.getSideProb(numCardsInSuit, 5)
                            j = 0
                            k =1
                            for i in range(11, 15):
                                if CardHelper.has_card(myHand,CardHelper.to_card(suit, i)):
                                    specialProb *= 1
                                    k+=1
                                else:
                                    specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                                52 - self.tricksInRound - j)
                                    j += 1
                                specialProb *= self.getSideProb(numCardsInSuit, k)
                        elif rank == 9:
                            j = 0
                            k = 1
                            for i in range(10, 15):
                                if CardHelper.has_card(myHand,CardHelper.to_card(suit, i)):
                                    specialProb *= 1
                                    k+=1
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
            sideSuitCuts = {0:1,
                       1:1,
                       2:1,
                       3:1}
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
                    for suit in [0,1,2,3]:
                        if suit != self.trumpSuit:
                            k = mySuits[suit][0]
                            i = sideSuitCuts[suit]
                            prob = self.getSideProb(k,k+i)
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
            minTrumpBid = 1.2*mySuits[self.trumpSuit][0]**2/self.tricksInRound
            if minTrumpBid > self.tricksInRound/2:
                minTrumpBid = floor(self.tricksInRound/2)
            if trumpBids < minTrumpBid:
                trumpBids = minTrumpBid
            # predBid is our predicted bid.
            # This is without considering other players bids or the position in which we bid
            predBid = trumpBids + expectedBid
            if p != 0:
                # Weighting by other players bids
                predBid -= (sum(self.bids) - (self.tricksInRound/self.numberOfPlayers * p))/(self.numberOfPlayers-p)
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
            if p == self.numberOfPlayers-1:
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
                    if round(predBid) == 0:
                        self.bids[startingPlayer] = round(predBid) + 1
                        bidsInPlayerOrder[p] = round(predBid) + 1
                    elif sum(self.bids) == self.tricksInRound:
                        self.bids[startingPlayer] = round(predBid) + 1
                        bidsInPlayerOrder[p] = round(predBid) + 1
                    elif sum(self.bids)+predBid -self.tricksInRound >= 0:
                        self.bids[startingPlayer] = round(predBid)+1
                        bidsInPlayerOrder[p] = round(predBid)+1
                    else:
                        self.bids[startingPlayer] = round(predBid) - 1
                        bidsInPlayerOrder[p] = round(predBid) - 1
            else:
                self.bids[startingPlayer] = round(predBid)
                bidsInPlayerOrder[p] = round(predBid)

            # Sanity check to make sure that if we have a really high trump, we bid at least 1
            countHighTrump = 0
            for suit, rank in CardHelper.iter_cards(mySuits[self.trumpSuit][2]):
                if rank >10:
                    countHighTrump+=1
            if countHighTrump > 0 and self.bids[startingPlayer] == 0:
                self.bids[startingPlayer] +=1
                bidsInPlayerOrder[p] = self.bids[startingPlayer]
                if sum(self.bids)==self.tricksInRound:
                    self.bids[startingPlayer] += 1
                    bidsInPlayerOrder[p] += 1

            # Another sanity check to make sure that we have at enough winners to bid this amount
            count = 0
            for suit, rank in CardHelper.iter_cards(myHand):
                if suit == self.trumpSuit:
                    count +=1
                if self.tricksInRound >= 8:
                    if rank > 11:
                        count +=1
                elif self.tricksInRound >= 6:
                    if rank > 10:
                        count +=1
                elif self.tricksInRound >= 4:
                    if rank > 9:
                        count +=1
                elif self.tricksInRound >= 2:
                    if rank > 8:
                        count +=1
                else:
                    count = 0
            if count < self.bids[startingPlayer]:
                if sum(self.bids) -1 != self.tricksInRound:
                    self.bids[startingPlayer] -= 1
            if verbose:
                print("PRED BID:", predBid, "Player:", startingPlayer)
                print("ACTUAL BID:", self.bids[startingPlayer], "Player:", startingPlayer, "\n")
        if self.bids[startingPlayer] < 0:
            if sum(self.bids) + 1 == self.tricksInRound:
                self.bids[startingPlayer] = 1
            else:
                self.bids[startingPlayer] = 0
        if not all(self.isAI):
            print("Player",startingPlayer, "Bid", self.bids[startingPlayer])
    def getUserCards(self,player):
        cards = []
        print("Hello player "+str(player) + "! Enter your cards below\n\n")
        for i in range(1,self.tricksInRound+1):
            bad = True
            while bad:
                card = input("Please enter card "+str(i)+": ")
                cardSuit = card[1]
                cardRank = card[0]
                if cardRank == "A":
                    cardRank = 14
                elif cardRank == "K":
                    cardRank = 13
                elif cardRank == "Q":
                    cardRank = 12
                elif cardRank == "J":
                    cardRank = 11
                elif cardRank == "T":
                    cardRank = 10
                else:
                    if self.checkInput(cardRank):
                        cardRank = int(card[0])
                if cardRank not in range(2, 14 + 1) or cardSuit not in ['C', 'D', 'H', 'S']:
                    print("Please enter a valid Card")
                else:
                    bad = False
            cards.append(CardHelper.to_card(cardSuit,cardRank))
        return CardHelper.list_to_hand(cards)
    def checkInput(self,val):
        try:
            int(val)
            return True
        except ValueError:
            return False
    def Deal(self):
        """ Reset the game state for the beginning of a new round, and deal the cards.
        """
        self.playerToMove = self.GetNextPlayer(self.dealer)
        self.discards = 0
        self.currentTrick = []
        self.tricksTaken = [0 for p in range(0, self.numberOfPlayers)]
        self.bids = [0 for p in range(0, self.numberOfPlayers)]
        deck = CardHelper.to_list(self.GetCardDeck())
        deckMask = self.GetCardDeck()
        random.shuffle(deck)
        trumpCard = 0
        if any(self.enterCards):
            bad = True
            while bad:
                card = input("Please enter the trump card: ")
                cardSuit = card[1]
                cardRank = card[0]
                if cardRank == "A":
                    cardRank = 14
                elif cardRank == "K":
                    cardRank = 13
                elif cardRank == "Q":
                    cardRank = 12
                elif cardRank == "J":
                    cardRank = 11
                elif cardRank == "T":
                    cardRank = 10
                else:
                    if self.checkInput(cardRank):
                        cardRank = int(card[0])
                if cardRank not in range(2, 14 + 1) or cardSuit not in ['C', 'D', 'H', 'S']:
                    print("Please enter a valid Card")
                else:
                    bad = False
            suitMap = {'C':0, 'D':1, 'H':2, 'S':3}
            trumpCard = CardHelper.to_card(cardRank,suitMap[cardSuit])
            deck.remove(trumpCard)
            deckMask = CardHelper.remove_card(deckMask,trumpCard)
        for p in range(0, self.numberOfPlayers):
            if self.enterCards[p]:
                self.playerHands[p] = self.getUserCards(p)
                deckMask = CardHelper.remove_hand(deckMask,self.playerHands[p])
                deck = CardHelper.to_list(deckMask)
        for p in range(0, self.numberOfPlayers):
            if not self.enterCards[p]:
                if any(self.enterCards):
                    self.playerHands[p] = deckMask
                else:
                    self.playerHands[p] = CardHelper.list_to_hand(deck[: self.tricksInRound])
                    # deckMask = CardHelper.remove_hand(deckMask,self.playerHands[p])
                    deck = deck[self.tricksInRound :]
        if not any(self.enterCards):
            self.flippedOverCard = deck[0]
        else:
            self.flippedOverCard = trumpCard
        self.trumpSuit = CardHelper.get_card_suit(self.flippedOverCard,isHand=False)
        print("FlippedOverCard: ",CardHelper.to_str(self.flippedOverCard))
    def GetNextPlayer(self, p):
        """ Return the player to the left of the specified player
        """
        if p == self.numberOfPlayers - 1:
            next = 0
        else:
            next = p + 1
        return next

    def onTrickWin(self,winner):
        for observer in range(self.numberOfPlayers):
            if self.useHeuristic[observer]:
                if observer != winner:
                    for c in range(52):
                        prob = self.probTables[observer][winner][c]
                        if prob > 1e-5:
                            rankfactor = 1.2-CardHelper.get_card_rank(c,isHand=False)/40
                            suit = CardHelper.get_card_suit(c,isHand=False)
                            if suit == self.trumpSuit:
                                suitfactor = 0.88
                            else:
                                suitfactor = 1
                            needed = self.GetTricksNeeded(winner)
                            tricksfactor = 1+ needed/25
                            factor = rankfactor*suitfactor*tricksfactor
                            # print(factor,self.probTables[observer][winner][c])
                            change = self.probChange(factor, self.probTables[observer][winner][c])
                            self.probTables[observer][winner][c] += change
                            self.adjustProbabilities(observer, self.playerToMove, c, change)
    def updateProbTables(self,move,leadSuit):
        """
        What things impact a players perception of the other players hands?
        - A player who needs 0 tricks AND ISN'T LEADING - we know they have no cards with higher rank than the one they played THAT WOULDNT WIN TRICK
            - If they are leading and they play a high ish card, we can likely guess they have no other cards in that suit. WEIGHT this by number of tricks left. More tricks left = less likely to be last card.
            - If they play a trump card, likely have no other trumps (or a really low trump)
            - If a player who needs 0 tricks WINS a trick, they probably have no choice (no lower cards of suit)
        - A player who doesn't follow suit and doesn't play a trump card likely has no other cards in that suit. Can weight this based on the rank of the card played and the number of tricks they need.
        - A player needs one trick and wins the trick. Likely has no cards of higher rank in that suit
        """
        winningCard, winningPlayer = self.getWinnerMidTrick()
        rank = CardHelper.get_card_rank(move,isHand=False)
        suit = CardHelper.get_card_suit(move,isHand=False)
        for observer in range(self.numberOfPlayers):
            if self.useHeuristic[observer]:
                # if not all([self.probTables[observer][self.playerToMove][i] <= 1 for i in range (52)]):
                #     print(observer,self.playerToMove,move,leadSuit)
                #     self.printProbTables(self.probTables,observer,self.playerToMove)
                if observer == self.playerToMove:
                    self.onCardPlayed(move,observer)
                else:
                    self.onCardPlayed(move, observer)
                    self.onVoid(move,observer,leadSuit)
                    for p in self.players:
                        assert not all(self.voids[s][p] for s in range(4))
                    tricksNeeded = self.GetTricksNeeded(self.playerToMove)
                    if tricksNeeded == 0:
                        if self.currentTrick[0][0] == self.playerToMove and rank > 7:
                            self.onNoTrickLead(suit,rank,observer)
                            # IF HIGH ISH CARD (7 or higher???), LIKELY HAVE NO CARDS LEFT OF THAT SUIT
                            # WEIGHT BY CARDS LEFT -> FEWER CARDS LEFT -> MORE LIKELY NO CARDS HIGHER
                        else:
                            if winningCard == move and self.currentTrick[0][0] != self.playerToMove:
                                self.onNoTrickWinning(suit,rank,observer,self.numberOfPlayers-len(self.currentTrick))
                                # THEY PROBABLY HAVE NO CARDS LOWER THAN THIS IN THIS SUIT
                            else:
                                self.onNoTrickLosing(suit,rank,observer,winningCard)
                                # IF TRUMP SUIT:
                                #   NO CARDS WITH HIGHER RANK
                                #   DECREASE LIKELIHOOD OF CARDS LOWER THAN MOVE
                                # ELSE:
                                #   NO CARDS WITH HIGHER RANK THAT WOULD STILL LOSE
                    elif tricksNeeded == 1 and winningPlayer == self.playerToMove:
                        self.onOneTrickWinning(suit,rank,observer)
                        # NO CARDS OF HIGHER RANK IN SUIT
                    else:
                        if suit != CardHelper.get_card_suit(self.currentTrick[0][1],isHand=False) and suit != self.trumpSuit:
                            self.onShortSuitNoTrump(suit,rank,observer,tricksNeeded)
                        # DOESNT FOLLOW SUIT AND NOT A TRUMP CARD LIKELY DOESNT HAVE ANY MORE OF THAT SUIT
                        # WEIGHT BY RANK, TRICKS NEEDED, AND TRICKSLEFT
                        # HIGH RANK -> SLIGHTLY LIKELIER NO OTHER CARDS IN SUIT
                        # MORE TRICKS NEEDED -> EXTREMELY LIKELY NO OTHER CARDS IN SUIT (they wouldve trumped in otherwise)
                        # MORE CARDS LEFT -> LESS LIKELY NO OTHER CARDS IN SUIT
        pass
    def onShortSuitNoTrump(self,suit,rank,observer,tricksNeeded):
        card = CardHelper.to_card(suit, rank)
        tricksLeft = CardHelper.get_num_cards(self.playerHands[self.playerToMove])
        for c in range(suit*13,(suit+1)*13):
            if self.probTables[observer][self.playerToMove][c] > 1e-5:
                highRankFactor = (2.8/sqrt(rank))
                if tricksNeeded > 3:
                    tricksNeededFactor = 0.3
                elif tricksNeeded > 2:
                    tricksNeededFactor = 0.5
                else:
                    tricksNeededFactor = 1
                tricksLeftFactor = tricksLeft/3
                factor = highRankFactor*tricksNeededFactor*tricksLeftFactor
                if factor > 2:
                    factor = 1.5

                change = self.probChange(factor,self.probTables[observer][self.playerToMove][c])
                self.probTables[observer][self.playerToMove][c] += change
                self.adjustProbabilities(observer,self.playerToMove,c,change)
    def onOneTrickWinning(self,suit,rank,observer):
        card = CardHelper.to_card(suit, rank)
        for c in range(card + 1, (suit + 1) * 13):
            if self.probTables[observer][self.playerToMove][c] > 1e-5:
                change = self.probChange(1/(3*(CardHelper.get_card_rank(c,isHand=False)-rank)),self.probTables[observer][self.playerToMove][c])
                self.probTables[observer][self.playerToMove][c] += change
                self.adjustProbabilities(observer, self.playerToMove, c, change)
    def onNoTrickLosing(self,suit,rank,observer,winningCard):
        card = CardHelper.to_card(suit,rank)
        if suit == self.trumpSuit:
            for c in range(card + 1, (suit + 1) * 13):
                if self.probTables[observer][self.playerToMove][c] > 1e-5:
                    change = -self.probTables[observer][self.playerToMove][c]
                    self.probTables[observer][self.playerToMove][c] += change
                    self.adjustProbabilities(observer, self.playerToMove, c, change)
            # POTENTIALLY NO MORE LOWER TRUMPS?
            for c in range(suit*13,card):
                if self.probTables[observer][self.playerToMove][c] > 1e-5:
                    change = self.probChange(.5,self.probTables[observer][self.playerToMove][c])
                    self.probTables[observer][self.playerToMove][c] += change
                    self.adjustProbabilities(observer, self.playerToMove, c, change)
        else:
            for c in range(card+1, (suit+1)*13):
                if self.trickWinnerTable[winningCard][c][CardHelper.get_card_suit(self.currentTrick[0][1],isHand=False)][self.trumpSuit]:
                    if self.probTables[observer][self.playerToMove][c] > 1e-5:
                        change = -self.probTables[observer][self.playerToMove][c]
                        self.probTables[observer][self.playerToMove][c] += change
                        self.adjustProbabilities(observer, self.playerToMove, c, change)
                else:
                    break
    def onNoTrickWinning(self,suit,rank,observer,playersToGo):
        card = CardHelper.to_card(suit,rank)
        if playersToGo < 2:
            for c in range(suit*13,card):
                if self.probTables[observer][self.playerToMove][c] > 1e-5:
                    change = -self.probTables[observer][self.playerToMove][c]
                    self.probTables[observer][self.playerToMove][c] += change
                    self.adjustProbabilities(observer,self.playerToMove,c,change)
        else:
            for c in range(suit*13,card):
                if self.probTables[observer][self.playerToMove][c] > 1e-5:
                    # higher rank -> less likely they have cards left
                    # lower rank -> more likely they have cards left
                    change = self.probChange(2/(c % 13 + 2),self.probTables[observer][self.playerToMove][c])
                    self.probTables[observer][self.playerToMove][c] += change
                    self.adjustProbabilities(observer,self.playerToMove,c,change)
    def onNoTrickLead(self,suit,rank,observer):
        # If cards left > 6 and rank > 9, reduce chance of cards by a factor. Otherwise make them 0
        cardsLeft = CardHelper.get_num_cards(self.playerHands[self.playerToMove])+1
        card = CardHelper.to_card(suit,rank)
        for c in range(suit * 13, (suit + 1) * 13):
            if self.probTables[observer][self.playerToMove][c] > 1e-5:
                change = self.probChange(1/(1+CardHelper.get_card_rank(c,isHand=False)/13),self.probTables[observer][self.playerToMove][c])
                self.probTables[observer][self.playerToMove][c] += change
                self.adjustProbabilities(observer,self.playerToMove,c,change)
    def printProbTables(self,probTables,observer,opponent):
        s = []
        s.append("                       FROM PLAYER: "+str(observer)+" PERSPECTIVE:")
        for p in [opponent]:
            if p != observer:
                s.append("\n\n")
                if p == self.numberOfPlayers:
                    s.append("                 ------ Undealt Cards ------\n")
                else:
                    s.append("                ------ Player: " + str(p)+ " ------\n")
                s.append("     2     3     4     5     6     7     8     9     T     J     Q     K     A")
                for card in range(52):
                    if opponent != self.numberOfPlayers:
                        suit,rank = CardHelper.get_card_rank_suit(card,isHand=False)
                        ss = CardHelper.get_str_suit(suit)
                        cs = probTables[observer][p][card]
                        cs = f"{cs:.3f}"
                        if CardHelper.has_card(self.originalHands[observer],card):
                            cs = Fore.GREEN + cs+ Style.RESET_ALL
                        elif self.voids[suit][opponent]:
                            cs = Fore.RED + cs+ Style.RESET_ALL
                        elif card in [c for (player, c) in self.currentTrick]:
                            cs = Fore.BLUE + cs+ Style.RESET_ALL
                        elif card == self.flippedOverCard:
                            cs = Fore.YELLOW + cs+ Style.RESET_ALL
                        elif CardHelper.has_card(self.discards,card):
                            cs = Fore.MAGENTA + cs+ Style.RESET_ALL
                        else:
                            cs = cs
                        if rank == 2:
                            s.append("\n"+ss+"| ")

                        s.append(cs + " ")
                    else:
                        suit, rank = CardHelper.get_card_rank_suit(card, isHand=False)
                        ss = CardHelper.get_str_suit(suit)
                        cs = probTables[observer][p][card]
                        cs = f"{cs:.3f}"
                        if CardHelper.has_card(self.originalHands[observer], card):
                            cs = Fore.GREEN + cs + Style.RESET_ALL
                        elif card in [c for (player, c) in self.currentTrick]:
                            cs = Fore.BLUE + cs + Style.RESET_ALL
                        elif card == self.flippedOverCard:
                            cs = Fore.YELLOW + cs + Style.RESET_ALL
                        elif CardHelper.has_card(self.discards, card):
                            cs = Fore.MAGENTA + cs + Style.RESET_ALL
                        else:
                            cs = cs
                        if rank == 2:
                            s.append("\n" + ss + "| ")

                        s.append(cs + " ")
        print("".join(s))



    def onCardPlayed(self,move,observer):
        for j in range(self.numberOfPlayers+1):
            self.probTables[observer][j][move] = 0
    def onVoid(self,move,observer,leadSuit):
        if CardHelper.get_card_suit(move,isHand=False) != leadSuit and not self.voids[leadSuit][self.playerToMove]:
            self.voids[leadSuit][self.playerToMove] = True
            for card in range(leadSuit * 13, (leadSuit + 1) * 13):
                if self.probTables[observer][self.playerToMove][card] > 1e-5:
                    change = -self.probTables[observer][self.playerToMove][card]
                    self.probTables[observer][self.playerToMove][card] += change
                    self.adjustProbabilities(observer,self.playerToMove,card,change)
    def DoMove(self, move):
        """
        Performs the given move, and updates things as necessary. Particles get updated here as needed
        """
        self.currentTrick.append((self.playerToMove, move))
        self.playerHands[self.playerToMove] = CardHelper.remove_card(self.playerHands[self.playerToMove],move)
        # self.playerHands[self.playerToMove].remove(move)
        if self.main and self.tricksInRound!=1 and self.tricksInRound - sum(self.tricksTaken) > 1:
            # If this is the main OhHellState, update particles and regenerate if necessary
            # for ob in self.players:
            #     for p in self.players + [self.numberOfPlayers]:
            #         for c in range(52):
            #             if self.probTables[ob][p][c] > 1:
            #                 breakpoint()
            self.updateProbTables(move,CardHelper.get_card_suit(self.currentTrick[0][1],isHand=False))
        self.playerToMove = self.GetNextPlayer(self.playerToMove)
        # If we are at the end of a trick
        if any(True for (player, card) in self.currentTrick if player == self.playerToMove):
            # (leader, leadCard) = self.currentTrick[0]
            # suitedPlays = [(player, CardHelper.get_card_rank(card)) for (player, card) in self.currentTrick if CardHelper.get_card_suit(card) == CardHelper.get_card_suit(leadCard)]
            # trumpPlays = [(player, CardHelper.get_card_rank(card)) for (player, card) in self.currentTrick if CardHelper.get_card_suit(card) == self.trumpSuit]
            # # Sort cards based on rules of trick taking games
            # sortedPlays = (
            #         sorted(suitedPlays, key=lambda pair: pair[1]) +
            #         sorted(trumpPlays, key=lambda pair: pair[1])
            # )
            winner = self.currentTrick[0][0]
            card1 = self.currentTrick[0][1]
            leadSuit = CardHelper.get_card_suit(card1,isHand=False)
            for (player, card2) in self.currentTrick:
                if not self.trickWinnerTable[card1][card2][leadSuit][self.trumpSuit]:
                    winner = player
                    card1 = card2+0
            # Pick the winner and update stuff
            # trickWinner = sortedPlays[-1][0]
            trickWinner = winner
            if self.main and self.tricksInRound!=1:
                self.onTrickWin(trickWinner)
            self.tricksTaken[trickWinner] += 1
            for (player, card) in self.currentTrick:
                self.discards |= CardHelper.card_to_hand(card)
            self.currentTrick = []
            self.playerToMove = trickWinner
            if self.main:
                if not all(self.isAI):
                    print("Trick is over! Player",trickWinner,"won the trick. They have won",self.tricksTaken[trickWinner],"tricks and they need to win a total of",self.bids[trickWinner],"tricks.")
                    for player in self.players:
                        print("Player",player,"has won",self.tricksTaken[player],"tricks, needs",self.bids[player],"tricks.")
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        hand = self.playerHands[self.playerToMove]
        # If leading, play whatever you want
        if self.currentTrick == []:
            return hand
        else:
            (leader, leadCard) = self.currentTrick[0]
            # cardsInSuit = [card for card in hand if card.suit == leadCard.suit]
            cardsInSuit = CardHelper.get_cards_in_suit(CardHelper.get_card_suit(leadCard,isHand=False),hand)
            # follow suit if you can, if not play whatever
            if cardsInSuit != 0:
                return cardsInSuit
            else:
                return hand
    def GetScore(self, player):
        """
        Get the game result from the viewpoint of player using only their score. Normalized to be 1 or 0. Used in ISMCTS
        """
        if self.tricksTaken[player] == self.bids[player]:
            return 1
        else:
            return 0
    def GetActualScore(self, player):
        """
        Get the game result from the viewpoint of player. This is their actual score and will be used to print out and for playing game with humans
        """
        if self.tricksTaken[player] == self.bids[player]:
            return 10 + self.tricksTaken[player]
        else:
            return 0
    def GetTricksNeeded(self,player):
        return self.bids[player]-self.tricksTaken[player]
    def __repr__(self):
        """
        Return a human-readable representation of the state
        """
        # result = "Round %i" % self.round
        result = "Tricks: %i" % self.tricksInRound
        result += " | Player %i: " % self.playerToMove
        result += ", ".join(CardHelper.to_str(card) for card in CardHelper.to_list(self.playerHands[self.playerToMove]))
        result += " | Tricks: %i" % self.tricksTaken[self.playerToMove]
        result += " | Trump: %s" % CardHelper.get_str_suit(self.trumpSuit)
        result += " | Trick: ["
        result += ",".join(("%i:%s" % (player, CardHelper.to_str(card))) for (player, card) in self.currentTrick)
        result += "]"
        return result



def PlayRound(state,mcIterations,randomRollout,tricksterPlay):
    """
    Plays one round of Oh Hell.
    mcIterations is a list of the number of MC iterations to use for each player.
    and false meaning we incorporate other players scores into our score for ISMCTS.
    """
    # While moves are still left in the game
    while (state.GetMoves() != 0):
        start = time.perf_counter()
        player = state.playerToMove
        iterations = mcIterations[player]
        # If we only have one valid move, play it as that is our only choice
        if tricksterPlay[player]:
            tricksterBot = CSharpBot(state,player)
            m = tricksterBot.suggest_next_card()
            if verboseHuman:
                print(
                    "Player " + str(state.playerToMove) + " Plays " + str(CardHelper.to_str(m)))
            state.DoMove(m)
        else:
            if CardHelper.get_num_cards(state.GetMoves()) == 1:
                m = CardHelper.hand_to_card(state.GetMoves())
                if verboseHuman:
                    print(
                    "Player " + str(state.playerToMove) + " Plays " + str(CardHelper.to_str(m)))
                state.DoMove(m)
            elif state.GetTricksNeeded(player) < 0:
                # If we can't win, play the highest rank card you can
                m = CardHelper.get_highest_card(state.GetMoves())
                if verboseHuman:
                    print(
                        "Player " + str(state.playerToMove) + " Plays " + str(CardHelper.to_str(m)))
                state.DoMove(m)
            else:
                # If we aren't using ISMCTS
                if iterations == 0:
                    # If human player, prompt them to play a card
                    if not state.isAI[player]:
                        if verboseHuman:
                            state.printHand(player)
                        bad = True
                        while bad:
                            card = input("Type the card you want to play ")
                            move = CardHelper.str_to_card(card)
                            if CardHelper.has_card(state.GetMoves(),move):
                                bad = False
                            else:
                                print("Illegal move. Try again.")
                        state.DoMove(move)

                    else:
                        m = random.choice(CardHelper.to_list(state.GetMoves()))
                        if verboseHuman:
                            print(
                            "Player " + str(state.playerToMove) + " Plays " + CardHelper.to_str(m))
                        state.DoMove(m)
                else:
                    if state.currentTrick != []:
                        (leader, leadCard) = state.currentTrick[0]
                        hand = state.GetMoves()
                        cardsInSuit = CardHelper.get_cards_in_suit(CardHelper.get_card_suit(leadCard, isHand=False), hand)
                        if state.GetTricksNeeded(player) == 0 and cardsInSuit != 0:
                            m = CardHelper.get_highest_losing_card(hand, state.currentTrick, state.trumpSuit)
                            if m < 0:
                                m = ISMCTS(rootstate=state, itermax=iterations, verbose=False,
                                           randomRollout=randomRollout[player], mainPlayer=state.playerToMove)
                            if verboseHuman:
                                print(
                                    "Player " + str(state.playerToMove) + " Plays " + CardHelper.to_str(m))
                            state.DoMove(m)
                        else:
                            if iterations >= 100:
                                if verbose:
                                    state.printHand(state.playerToMove)
                            m = ISMCTS(rootstate=state, itermax=iterations, verbose=False,
                                       randomRollout=randomRollout[player], mainPlayer=state.playerToMove)
                            if verboseHuman:
                                print(
                                    "Player " + str(state.playerToMove) + " Plays " + CardHelper.to_str(m))
                            state.DoMove(m)
                    else:
                        m = ISMCTS(rootstate=state, itermax=iterations,verbose=False,randomRollout=randomRollout[player],mainPlayer=state.playerToMove)
                        if verboseHuman:
                            print(
                            "Player " + str(state.playerToMove) + " Plays "+CardHelper.to_str(m))
                        state.DoMove(m)
        end = time.perf_counter()-start
        # if iterations != 2000 and state.tricksInRound != 1:
        #     state.printProbTables(state.probTables,0,player)
        # if state.tricksInRound != 1 and not state.isAI[player]:
        #     state.printHand(player)
        # print("TIME ELAPSED IN SECONDS: ",end)
    tempscores = []
    for p in range(0, state.numberOfPlayers):
        score = state.GetActualScore(p)
        tempscores.append(score)
        print(
        "Player " + str(p) + " scores: " + str(score) + "\n")
    return tempscores

def playFullGame(numPlayers,dealer,main,isAI,useHeuristic,bidStyle,enterCards,mcIterations,randomRollout,tricksterPlay,tricksterBid):
    tricks = [x for x in range(10, 0, -1)] + [x for x in range(2, 11)]
    # tricks = [1]
    scoreboard = [0 for i in range(numPlayers)]
    scores = []
    for numTricks in tricks:
        print("DEALER: " + str(dealer))
        state = OhHellState(numPlayers,numTricks,dealer,main,isAI,useHeuristic,bidStyle,enterCards,tricksterBid)
        roundScores = PlayRound(state, mcIterations,randomRollout,tricksterPlay)
        scores.append(roundScores)
        dealer = state.GetNextPlayer(dealer)
        for p in state.players:
            scoreboard[p] += state.GetActualScore(p)
        print("SCOREBOARD:")
        for p in state.players:
            print("Player " + str(p) + ": " + str(scoreboard[p]))
        print("")
        # print("Player One:", scoreboard[0])
        # print("Player Two:", scoreboard[1])
        # print("Player Three:", scoreboard[2])
        # print("Player Four:", scoreboard[3])
        # print("Player Five:", scoreboard[4], "\n")
    return scoreboard,scores

if __name__ == "__main__":
    allTimeWins = [0,0,0,0]
    allTimeScore = [0,0,0,0]
    scoreList = []
    dealer = 3
    data = {
        "seed": [],
        "winner": [],
        "score1": [],
        "score2": [],
        "score3": [],
        "score0": [],
        "dealer": [],
        "round1":[],
        "round2": [],
        "round3": [],
        "round4": [],
        "round5": [],
        "round6": [],
        "round7": [],
        "round8": [],
        "round9": [],
        "round10": [],
        "round11": [],
        "round12": [],
        "round13": [],
        "round14": [],
        "round15": [],
        "round16": [],
        "round17": [],
        "round18": [],
        "round19": [],
    }
    count = 4
    for seed in range(37000,200000):
        start = time.perf_counter()
        print(seed)
        random.seed(seed)
        mcIterations = [1000,250,100,10]
        scoreboard,scores = playFullGame(4, dealer, True, [True, True, True, True], [True,True,True,True],["normal", "normal", "normal", "normal"],[False,False,False,False],mcIterations,[True,True,True,True])
        # mcIterations = [1000,250,100,0]
        # scores = playFullGame(4, dealer, True, [True, True, True, False], [True,True,True,False],["normal", "normal", "normal", "normal"],[False,False,False,False],mcIterations,[True,True,True,False])
        winner = 0
        scoreList.append(scoreboard)
        highScore = scoreboard[0]
        for i in range(4):
            allTimeScore[i] += scoreboard[i]
            if scoreboard[i] >= highScore:
                winner = i
                highScore = scoreboard[i]
        allTimeWins[winner] += 1
        data["dealer"].append(dealer)
        if dealer == 3:
            dealer = 0
        else:
            dealer += 1
        data["seed"].append(seed)
        data["winner"].append(winner)
        data["score1"].append(scoreboard[1])
        data["score0"].append(scoreboard[0])
        data["score2"].append(scoreboard[2])
        data["score3"].append(scoreboard[3])
        data["round1"].append(scores[0])
        data["round2"].append(scores[1])
        data["round3"].append(scores[2])
        data["round4"].append(scores[3])
        data["round5"].append(scores[4])
        data["round6"].append(scores[5])
        data["round7"].append(scores[6])
        data["round8"].append(scores[7])
        data["round9"].append(scores[8])
        data["round10"].append(scores[9])
        data["round11"].append(scores[10])
        data["round12"].append(scores[11])
        data["round13"].append(scores[12])
        data["round14"].append(scores[13])
        data["round15"].append(scores[14])
        data["round16"].append(scores[15])
        data["round17"].append(scores[16])
        data["round18"].append(scores[17])
        data["round19"].append(scores[18])
        end = time.perf_counter()
        elapsed = end - start
        print(elapsed)
        if elapsed > 900:
            print("yeet")

            start = time.perf_counter()
            df = pd.DataFrame(data)
            df.to_csv(f"outFourAIFullGame{count}HrOne.csv", index=False)
            data = {
                "seed": [],
                "winner": [],
                "score1": [],
                "score2": [],
                "score3": [],
                "score0": [],
                "dealer": [],
                "round1": [],
                "round2": [],
                "round3": [],
                "round4": [],
                "round5": [],
                "round6": [],
                "round7": [],
                "round8": [],
                "round9": [],
                "round10": [],
                "round11": [],
                "round12": [],
                "round13": [],
                "round14": [],
                "round15": [],
                "round16": [],
                "round17": [],
                "round18": [],
                "round19": [],
            }
            count += 1
        print(scoreList)
        print(allTimeWins)
        print(allTimeScore)