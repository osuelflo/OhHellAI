from math import *
import random, sys
from copy import deepcopy

import numpy as np
import pandas as pd
verbose = False


class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement ISMCTS in any imperfect information game,
        although they could be enhanced and made quicker, for example by using a
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1, 2, ..., self.numberOfPlayers.
    """

    def __init__(self):
        self.numberOfPlayers = 4
        self.playerToMove = 1

    def GetNextPlayer(self, p):
        """ Return the player to the left of the specified player
        """
        return (p % self.numberOfPlayers) + 1

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = GameState()
        st.playerToMove = self.playerToMove
        return st

    def CloneAndRandomize(self, observer):
        """ Create a deep clone of this game state, randomizing any information not visible to the specified observer player.
        """
        return self.Clone()

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        self.playerToMove = self.GetNextPlayer(self.playerToMove)

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        pass

    def GetResult(self, player):
        """ Get the game result from the viewpoint of player.
        """
        pass

    def __repr__(self):
        """ Don't need this - but good style.
        """
        pass


class Card:
    """ A playing card, with rank and suit.
        rank must be an integer between 2 and 14 inclusive (Jack=11, Queen=12, King=13, Ace=14)
        suit must be a string of length 1, one of 'C' (Clubs), 'D' (Diamonds), 'H' (Hearts) or 'S' (Spades)
    """

    def __init__(self, rank, suit):
        if rank not in range(2, 14 + 1):
            raise Exception("Invalid rank")
        if suit not in ['C', 'D', 'H', 'S']:
            raise Exception("Invalid suit")
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return "??23456789TJQKA"[self.rank] + self.suit

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __ne__(self, other):
        return self.rank != other.rank or self.suit != other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

class OhHellState(GameState):
    def __init__(self, n,numTricks,numParticles,lamb,dealer,main,isAI,useHeuristic,bidStyle,enterCards,start = True):
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
        super().__init__()
        # Initializing all the class variables
        self.enterCards = enterCards
        self.bidStyle = bidStyle
        self.isAI = isAI
        self.main = main
        self.numParticles = numParticles
        self.useHeuristic = useHeuristic
        self.lamb = log(lamb)
        self.players = list(range(0,n))
        self.dealer = dealer
        self.flippedOverCard = None
        self.numberOfPlayers = n
        self.ESS = [self.numParticles for p in range(self.numberOfPlayers)]
        # print(self.ESS, self.numberOfPlayers)
        self.playerToMove = 0
        self.tricksInRound = numTricks
        self.playerHands = [[] for p in range(0, self.numberOfPlayers)]
        self.discards = []  # Stores the cards that have been played already in this round
        self.currentTrick = []
        self.trumpSuit = None
        self.tricksTaken = []  # Number of tricks taken by each player this round
        # self.knockedOut = {p: False for p in range(1, self.numberOfPlayers + 1)}
        self.sideSuitProbs,self.trumpSuitProbs = self.getProbTables()
        self.sideOneTrickProbs, self.trumpOneTrickProbs = self.getOneTrickProbsTable()
        self.bids = []
        self.particles = []
        self.logWeights = []
        self.weights = []
        self.voids = self.createVoids()
        if start:
            self.Deal()
            if not all(self.isAI):
                print("Welcome to a game of Oh Hell!")
                print("\nThe flipped over card is", self.flippedOverCard)
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
                    self.Bid(bidsInPlayerOrder,p,startingPlayer)
            # If we only have one card, that is the only move we can make so must play it.
            if self.tricksInRound != 1:
                self.particles,self.logWeights,self.weights = self.initializeParticles(numParticles)
                self.makeBidInferences()
        # These values help normalize the reward when using the GetReward function that incorporates other players' scores
        self.minRes = -((self.numberOfPlayers - 1)*10 + self.tricksInRound)/self.numberOfPlayers
        self.maxRes = 10+self.tricksInRound
        self.range = self.maxRes - self.minRes
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
        voids = {"C": [False for p in range(0, self.numberOfPlayers)],
                 "D": [False for p in range(0, self.numberOfPlayers)],
                 "S": [False for p in range(0, self.numberOfPlayers)],
                 "H": [False for p in range(0, self.numberOfPlayers)]}
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
        return [card for card in self.GetCardDeck() if
                       card in self.discards or card in self.playerHands[
                           observer] or card in [c for (player, c) in self.currentTrick] or card == self.flippedOverCard]
    def getUnseenCards(self,observer):
        """
        Gets all cards that the observer has not seen. Note that not all cards are actually dealt.
        """
        return [card for card in self.GetCardDeck() if
                card not in self.discards and card not in self.playerHands[
                    observer] and card not in [c for (player, c) in self.currentTrick] and card != self.flippedOverCard]
    def systematic_resample(self):
        """
        Uses systematic resampling to resample from the set of valid particles (weight != 0) if ESS < 0.3*N
        """
        for p in range(self.numberOfPlayers):
            if self.ESS[p] < 0.3*self.numParticles:
                # Getting valid particles and then making new uniform weights
                self.ESS[p] = 1.0
                valid_indices = [i for i, w in enumerate(self.weights[p]) if w > 0.0]
                new_weights = [1 / self.numParticles for j in range(self.numParticles)]
                # Doing systematic resampling
                w = np.array([self.weights[p][i] for i in valid_indices], dtype=float)
                w /= np.sum(w)
                cdf = np.cumsum(w)
                u0 = np.random.rand() / self.numParticles
                positions = u0 + np.arange(self.numParticles) / self.numParticles
                idx = np.searchsorted(cdf, positions)
                new_particles = [
                    deepcopy(self.particles[p][valid_indices[i]])
                    for i in idx
                ]
                # Updating class variables
                self.weights[p] = new_weights
                self.logWeights[p] = [log(w) for w in new_weights]
                self.particles[p] = new_particles
    def regenerate(self,observer):
        """
        Regenerating particles if all particles have weight of 0.
        New particles are generated that are consistent with the observer's seen cards.
        They also incorporate the other players' perceived strength by running the bidding algorithm on each hand generated,
        and weighting the particle based on the difference between the predicted tricks won (bid) vs. the amount of tricks
        the player ACTUALLY still has to win in the game
        """
        new_particles = [
            [[] for i in self.players]
            for j in range(self.numParticles)
        ]
        unseen = self.getUnseenCards(observer)
        for part in range(self.numParticles):
            badDeal = True
            # It is possible for a given random deal to not be feasible given the cards that have already been played
            # and if lots of players are void in suits. So we need to keep trying until a valid deal is reached
            while badDeal:
                tempUnseen = deepcopy(unseen)
                random.shuffle(tempUnseen)
                players = deepcopy(self.players)
                random.shuffle(players)
                for play in players:
                    tempHand = []
                    # If the player is the observer, they know their own hand
                    if play == observer:
                        new_particles[part][play] = deepcopy(self.playerHands[play])
                    else:
                        count = 0
                        # start with the first player and deal them cards. If the card's suit is a suit they are void in,
                        # add it back to the pile of undealt cards and try a different card
                        while len(tempHand)<len(self.playerHands[play]) and len(tempUnseen) != 0 and count < len(unseen)*self.numberOfPlayers:
                            newCard = tempUnseen.pop()
                            if self.voids[newCard.suit][play]:
                                tempUnseen.append(newCard)
                            else:
                                tempHand.append(newCard)
                            count +=1
                        new_particles[part][play] = tempHand
                # checking if any of the players dont have enough cards, which means it is a bad deal
                badDeal = any([len(new_particles[part][i]) != len(self.playerHands[i]) for i in self.players])
        self.particles[observer] = new_particles
        self.logWeights[observer] = [log(1/self.numParticles) for j in range(self.numParticles)]
        self.normalizeWeights(observer)
        self.makeRegenBidInferences(observer)
    def normalizeWeights(self,observer):
        """
        Normalizing weights for the observer so they sum to 1
        """
        logW = self.logWeights[observer]
        maxLogW = max(logW)
        shifted = [w - maxLogW for w in logW]
        weights = [exp(w) if w > -1e300 else 0.0 for w in shifted]
        total = sum(weights)
        assert total > 0.0
        weights = [w / total for w in weights]
        self.weights[observer] = weights
        self.ESS[observer] = 1.0 / sum(w * w for w in weights)
    def makeRegenBidInferences(self,player):
        """
        Helper function to make the bid inferences during the particle regeneration phase.
        If a player has won more tricks than they bid, we reset their "tricks to win" variable to 0.
        Then, update the logProbabilities to punish particles with poor expected tricks to win vs. the real amount,
        weighted by the parameter lambda
        """
        particleSet = self.particles[player]
        for j in range(len(particleSet)):
            game = particleSet[j]
            predBids = self.inferenceBid(game, player)
            delta = []
            for i in range(self.numberOfPlayers):
                tricksToWin = self.bids[i] - self.tricksTaken[i]
                if tricksToWin < 0:
                    tricksToWin = 0
                delta.append(abs(predBids[i] - tricksToWin))
            newDelta = sum(delta)
            self.logWeights[player][j] += -self.lamb * newDelta
    def makeBidInferences(self):
        """
        Helper function to make the bid inferences at the beginning of the game after each player has bid.
        Then, update the logProbabilities to punish particles with poor expected tricks to win vs. the real amount,
        weighted by the parameter lambda
        """
        for player in range(self.numberOfPlayers):
            if self.isAI[player] and self.useHeuristic[player]:
                particleSet = self.particles[player]
                for j in range(len(particleSet)):
                    game = particleSet[j]
                    predBids = self.inferenceBid(game,player)
                    delta = sum([abs(predBids[i]-self.bids[i]) for i in range(self.numberOfPlayers)])
                    self.logWeights[player][j] += -self.lamb*delta
                self.normalizeWeights(player)
        self.systematic_resample()
    def initializeParticles(self,numParticles = 5000):
        """
        Creates a numPlayers x numPlayers x numParticles particles array.
        For example, self.particles[0][3][100] would be the hand dealt to player 3 from player 0s perspective in particle 100
        """
        particles = []
        logweights = []
        weights = []
        for p in range(self.numberOfPlayers):
            temp = []
            temp3 = []
            temp4 = []
            if self.useHeuristic[p]:
                for part in range(numParticles):
                    temp2 = []
                    unseenCards = [card for card in self.GetCardDeck() if card not in self.playerHands[p] and card != self.flippedOverCard]
                    random.shuffle(unseenCards)
                    for p2 in range(self.numberOfPlayers):
                        if p2 == p:
                            temp2.append(deepcopy(self.playerHands[p2]))
                        else:
                            temp2.append(unseenCards[:self.tricksInRound])
                            unseenCards = unseenCards[self.tricksInRound:]
                    temp.append(temp2)
                    temp3.append(log(1 / numParticles))
                    temp4.append(1 / numParticles)
                particles.append(temp)
                logweights.append(temp3)
                weights.append(temp4)
            else:
                particles.append(temp)
                logweights.append(temp3)
                weights.append(temp4)
        return particles,logweights,weights
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
        return pd.read_csv(f"probabilityData/{self.tricksInRound}Tricks{self.numberOfPlayers}PSideSuit.csv"), pd.read_csv(f"probabilityData/{self.tricksInRound}Tricks{self.numberOfPlayers}PTrumpSuit.csv")
    def Clone(self):
        """
        Create a deep clone of this game state.
        Notably does not clone anything related to particle filtering as this information is not needed for
        game states solely used for ISMCTS simulation.
        """
        st = OhHellState(self.numberOfPlayers,self.tricksInRound,self.numParticles,self.lamb,self.dealer,False,self.isAI,self.useHeuristic,self.bidStyle,self.enterCards,False)
        st.enterCards = self.enterCards
        st.isAI = self.isAI
        st.bidStyle = self.bidStyle
        st.useHeuristic = self.useHeuristic
        st.main = False
        st.numParticles = self.numParticles
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
        st.lamb = self.lamb
        st.minRes = -((self.numberOfPlayers - 1) * 10 + self.tricksInRound) / 4
        st.maxRes = 10 + self.tricksInRound
        st.range = self.maxRes - self.minRes
        st.voids = deepcopy(self.voids)
        return st
    def inferenceBid(self, particle,observer):
        """
        Bidding algorithm specifically for particles. More detailed commenting can be found in the Bid() function
        """
        inferredBids = [0 for j in range(0, self.numberOfPlayers)]
        startingPlayer = self.dealer
        bidsInPlayerOrder = [0 for j in range(0, self.numberOfPlayers)]
        for p in range(0, self.numberOfPlayers):
            expectedBid = 0
            startingPlayer = self.GetNextPlayer(startingPlayer)
            if startingPlayer == observer:
                inferredBids[observer] = self.bids[observer]
                bidsInPlayerOrder[p] = self.bids[observer]
            else:
                myHand = particle[startingPlayer]
                if self.tricksInRound == 1:
                    if p == self.numberOfPlayers - 1 and sum(inferredBids) == 0:
                        inferredBids[startingPlayer] = 0
                        bidsInPlayerOrder[p] = 0
                    elif p == self.numberOfPlayers - 1 and sum(inferredBids) == 1:
                        inferredBids[startingPlayer] = 1
                        bidsInPlayerOrder[p] = 1
                    else:
                        if sum(inferredBids) < 1:
                            if myHand[0].suit == self.trumpSuit:
                                inferredBids[startingPlayer] = round(
                                    self.getTrumpOneTrickProb(self.numberOfPlayers, p, myHand[0].rank) * (
                                                1 + p / (self.numberOfPlayers * 2)))
                                bidsInPlayerOrder[p] = round(
                                    self.getTrumpOneTrickProb(self.numberOfPlayers, p, myHand[0].rank))
                            else:
                                inferredBids[startingPlayer] = round(
                                    self.getSideOneTrickProb(self.numberOfPlayers, p, myHand[0].rank))
                                bidsInPlayerOrder[p] = round(
                                    self.getSideOneTrickProb(self.numberOfPlayers, p, myHand[0].rank))
                        else:
                            if sum(inferredBids) == 1 and bidsInPlayerOrder[0] == 0:
                                if myHand[0].suit == self.trumpSuit and myHand[0].rank >= 8:
                                    inferredBids[startingPlayer] = 1
                                    bidsInPlayerOrder[p] = 1
                            elif sum(inferredBids) == 1 and bidsInPlayerOrder[0] == 1:
                                if myHand[0].suit == self.trumpSuit and self.numberOfPlayers == 2:
                                    inferredBids[startingPlayer] = 1
                                    bidsInPlayerOrder[p] = 1
                                elif myHand[0].suit == self.trumpSuit and self.numberOfPlayers == 3:
                                    inferredBids[startingPlayer] = 1
                                    bidsInPlayerOrder[p] = 1
                                else:
                                    if myHand[0].suit == self.trumpSuit and myHand[0].rank >= 8:
                                        inferredBids[startingPlayer] = 1
                                        bidsInPlayerOrder[p] = 1
                            elif sum(inferredBids) == 2:
                                if bidsInPlayerOrder[0] == 0:
                                    if myHand[0].suit == self.trumpSuit and myHand[0].rank >= 11:
                                        inferredBids[startingPlayer] = 1
                                        bidsInPlayerOrder[p] = 1
                                else:
                                    if myHand[0].suit == self.trumpSuit and myHand[0].rank >= 8:
                                        inferredBids[startingPlayer] = 1
                                        bidsInPlayerOrder[p] = 1
                            elif sum(inferredBids) >= 3:
                                if myHand[0].suit == self.trumpSuit and myHand[0].rank >= 13:
                                    inferredBids[startingPlayer] = 1
                                    bidsInPlayerOrder[p] = 1
                else:
                    mySuits = {"C": [0, False, []],
                               "S": [0, False, []],
                               "H": [0, False, []],
                               "D": [0, False, []]}
                    mySuits[self.trumpSuit][1] = True
                    for card in myHand:
                        mySuits[card.suit][0] += 1
                        mySuits[card.suit][2].append(card)
                    for suit in ['C', 'S', 'H', 'D']:
                        if suit != self.trumpSuit:
                            numCardsInSuit = mySuits[suit][0]
                            for card in mySuits[suit][2]:
                                tempProb = 0
                                specialProb = 1
                                if card.rank == 14:
                                    tempProb = self.getSideProb(numCardsInSuit, 1)
                                    specialProb = 0
                                elif card.rank == 13:
                                    tempProb = self.getSideProb(numCardsInSuit, 2)
                                    j = 0
                                    k = 1
                                    for i in range(14, 15):
                                        if Card(i, suit) in myHand:
                                            specialProb *= 1
                                            k += 1
                                        else:
                                            specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                                        52 - self.tricksInRound - j)
                                            j += 1
                                        specialProb *= self.getSideProb(numCardsInSuit, k)
                                elif card.rank == 12:
                                    tempProb = self.getSideProb(numCardsInSuit, 3)
                                    j = 0
                                    k = 1
                                    for i in range(13, 15):
                                        if Card(i, suit) in myHand:
                                            specialProb *= 1
                                            k += 1
                                        else:
                                            specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                                    52 - self.tricksInRound - j)
                                            j += 1
                                        specialProb *= self.getSideProb(numCardsInSuit, k)
                                elif card.rank == 11:
                                    tempProb = self.getSideProb(numCardsInSuit, 4)
                                    j = 0
                                    k = 1
                                    for i in range(12, 15):
                                        if Card(i, suit) in myHand:
                                            specialProb *= 1
                                            k += 1
                                        else:
                                            specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                                    52 - self.tricksInRound - j)
                                            j += 1
                                        specialProb *= self.getSideProb(numCardsInSuit, k)
                                elif card.rank == 10:
                                    tempProb = self.getSideProb(numCardsInSuit, 5)
                                    j = 0
                                    k = 1
                                    for i in range(11, 15):
                                        if Card(i, suit) in myHand:
                                            specialProb *= 1
                                            k += 1
                                        else:
                                            specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                                    52 - self.tricksInRound - j)
                                            j += 1
                                        specialProb *= self.getSideProb(numCardsInSuit, k)
                                elif card.rank == 9:
                                    j = 0
                                    k = 1
                                    for i in range(10, 15):
                                        if Card(i, suit) in myHand:
                                            specialProb *= 1
                                            k += 1
                                        else:
                                            specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                                    52 - self.tricksInRound - j)
                                            j += 1
                                        specialProb *= self.getSideProb(numCardsInSuit, k)
                                if card.rank >= 9:
                                    if tempProb > specialProb:
                                        expectedBid += tempProb
                                    else:
                                        expectedBid += specialProb
                    sideSuitCuts = {"C": 1,
                                    "S": 1,
                                    "H": 1,
                                    "D": 1}
                    trumpCardProbs = [[] for x in range(mySuits[self.trumpSuit][0])]
                    count = 0
                    for trumpCard in sorted(mySuits[self.trumpSuit][2], key=lambda c: c.rank):
                        bestProb = 0
                        bestProbSuit = ""
                        if trumpCard.rank >= 10:
                            trumpCardProbs[count].append(1)
                        else:
                            for suit in ['C', 'S', 'H', 'D']:
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
                        trumpBids += max(c)
                    minTrumpBid = 1.2 * mySuits[self.trumpSuit][0] ** 2 / self.tricksInRound
                    if minTrumpBid > self.tricksInRound / 2:
                        minTrumpBid = floor(self.tricksInRound / 2)
                    if trumpBids < minTrumpBid:
                        trumpBids = minTrumpBid
                    predBid = trumpBids + expectedBid
                    if p != 0:
                        predBid -= (sum(inferredBids) - (self.tricksInRound / self.numberOfPlayers * p)) / (
                                    self.numberOfPlayers - p)
                    if predBid < 0:
                        predBid = 0
                    if self.numberOfPlayers < 3:
                        predBid *= 1.3
                    elif self.numberOfPlayers >= 3:
                        predBid *= 1.13
                    if predBid > self.tricksInRound:
                        predBid = self.tricksInRound
                    if p == self.numberOfPlayers - 1:
                        if round(predBid) + sum(inferredBids) != self.tricksInRound:
                            if round(predBid) + sum(inferredBids) - self.tricksInRound > 2:
                                predBid = self.tricksInRound + 2 - sum(inferredBids)
                                inferredBids[startingPlayer] = round(predBid)
                                bidsInPlayerOrder[p] = round(predBid)
                            elif round(predBid) + sum(inferredBids) - self.tricksInRound < -2:
                                predBid = self.tricksInRound + 2 - sum(inferredBids)
                                inferredBids[startingPlayer] = round(predBid)
                                bidsInPlayerOrder[p] = round(predBid)
                            else:
                                inferredBids[startingPlayer] = round(predBid)
                                bidsInPlayerOrder[p] = round(predBid)
                        else:
                            if sum(inferredBids) == self.tricksInRound:
                                inferredBids[startingPlayer] = round(predBid) + 1
                                bidsInPlayerOrder[p] = round(predBid) + 1
                            elif sum(inferredBids) + predBid - self.tricksInRound > 0:
                                inferredBids[startingPlayer] = round(predBid) + 1
                                bidsInPlayerOrder[p] = round(predBid) + 1
                            else:
                                inferredBids[startingPlayer] = round(predBid) - 1
                                bidsInPlayerOrder[p] = round(predBid) - 1
                    else:
                        inferredBids[startingPlayer] = round(predBid)
                        bidsInPlayerOrder[p] = round(predBid)
        return inferredBids
    def CloneAndRandomize(self, observer):
        """ Create a deep clone of this game state, randomizing any information not visible to the specified observer player if they aren't using heuristic.
        If they are, then sample from the particles based on their weights.
        """
        st = self.Clone()
        if st.useHeuristic[observer]:
            # sample from particles
            seenCards = st.playerHands[observer] + st.discards + [st.flippedOverCard] + [card for (player, card) in
                                                                                         st.currentTrick]
            particle = deepcopy(random.choices(self.particles[observer], self.weights[observer], k =1))[0]
            for p in range(0,st.numberOfPlayers):
                st.playerHands[p] = particle[p]
            return st
        else:
            # The observer can see his own hand and the cards in the current trick, and can remember the cards played in previous tricks
            seenCards = st.playerHands[observer] + st.discards + [st.flippedOverCard] + [card for (player, card) in
                                                                                         st.currentTrick]
            # The observer can't see the rest of the deck
            unseenCards = [card for card in st.GetCardDeck() if card not in seenCards]

            # Deal the unseen cards to the other players
            random.shuffle(unseenCards)
            for p in range(0, st.numberOfPlayers):
                if p != observer:
                    # Deal cards to player p
                    # Store the size of player p's hand
                    numCards = len(self.playerHands[p])
                    # Give player p the first numCards unseen cards
                    st.playerHands[p] = unseenCards[: numCards]
                    # Remove those cards from unseenCards
                    unseenCards = unseenCards[numCards:]
            return st
    def GetCardDeck(self):
        """ Construct a standard deck of 52 cards.
        """
        return [Card(rank, suit) for rank in range(2, 14 + 1) for suit in ['C', 'D', 'H', 'S']]
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
        for card in self.playerHands[p]:
            cards_by_suit[card.suit].append(card)

        # Sort each suit by rank (high to low)
        for s in suit_order:
            if self.trumpSuit == s:
                suit_names[s] = suit_names[s]+" -- Trump"
            cards_by_suit[s].sort(key=lambda c: c.rank, reverse=True)

        # Print result
        st = "Player " + str(p) + ": "
        for s in suit_order:
            cards_str = " ".join(f"{"??23456789TJQKA"[card.rank]}{card.suit}" for card in cards_by_suit[s])
            st += "[" + cards_str + "]" + " "
        print(st)
    def Bid(self,bidsInPlayerOrder, p,startingPlayer):
        """
        Bidding algorithm used at the beginning of the game. p is the player order number
        (ex. if startingPlayer bids first, then p would be 0). startingPlayer is the player who is going to bid
        """
        expectedBid = 0
        myHand = deepcopy(self.playerHands[startingPlayer])
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
                    if myHand[0].suit == self.trumpSuit:
                        self.bids[startingPlayer] = round(self.getTrumpOneTrickProb(self.numberOfPlayers-p,0,myHand[0].rank))
                        bidsInPlayerOrder[p] = round(self.getTrumpOneTrickProb(self.numberOfPlayers-p,0,myHand[0].rank))
                    # Otherwise, look up probability. This will always be less than 0.5, so really doesn't matter
                    else:
                        self.bids[startingPlayer] = round(self.getSideOneTrickProb(self.numberOfPlayers, p, myHand[0].rank))
                        bidsInPlayerOrder[p] = round(
                            self.getSideOneTrickProb(self.numberOfPlayers, p, myHand[0].rank))
                else:
                    # If the first player didn't bid but someone else did, they must have a trump card
                    # (if acting rationally, which we assume). So we will also bid if we have a high trump
                    if sum(self.bids) == 1 and bidsInPlayerOrder[0] == 0:
                        if myHand[0].suit == self.trumpSuit and myHand[0].rank >= 8:
                            self.bids[startingPlayer] = 1
                            bidsInPlayerOrder[p] = 1
                    # If only the first player bid, we will also bid if we have a trump, though we might need
                    # a higher trump if there are more players in the game
                    elif sum(self.bids) == 1 and bidsInPlayerOrder[0] == 1:
                        if myHand[0].suit == self.trumpSuit and self.numberOfPlayers == 2:
                            self.bids[startingPlayer] = 1
                            bidsInPlayerOrder[p] = 1
                        elif myHand[0].suit == self.trumpSuit and self.numberOfPlayers == 3:
                            self.bids[startingPlayer] = 1
                            bidsInPlayerOrder[p] = 1
                        else:
                            if myHand[0].suit == self.trumpSuit and myHand[0].rank >= 8:
                                self.bids[startingPlayer] = 1
                                bidsInPlayerOrder[p] = 1
                    # If two (or more) players bid (very unlikely), then we better have a very high trump
                    elif sum(self.bids) == 2:
                        if bidsInPlayerOrder[0] == 0:
                            if myHand[0].suit == self.trumpSuit and myHand[0].rank >= 11:
                                self.bids[startingPlayer] = 1
                                bidsInPlayerOrder[p] = 1
                        else:
                            if myHand[0].suit == self.trumpSuit and myHand[0].rank >= 8:
                                self.bids[startingPlayer] = 1
                                bidsInPlayerOrder[p] = 1
                    elif sum(self.bids) >= 3:
                        if myHand[0].suit == self.trumpSuit and myHand[0].rank >= 13:
                            self.bids[startingPlayer] = 1
                            bidsInPlayerOrder[p] = 1
        else:
            # More than 1 trick in round
            # Counting the number of cards we have in each suit, and checking to see which one is trump suit
            mySuits = {"C":[0,False,[]],
                       "S":[0,False,[]],
                       "H":[0,False,[]],
                       "D":[0,False,[]]}
            mySuits[self.trumpSuit][1] = True
            for card in myHand:
                mySuits[card.suit][0] += 1
                mySuits[card.suit][2].append(card)
            for suit in ['C', 'S', 'H', 'D']:
                # Go through each suit and make bids with either two approaches: trump or off suit
                if suit != self.trumpSuit:
                    numCardsInSuit = mySuits[suit][0]
                    for card in mySuits[suit][2]:
                        # Go through each card in the off suit, and if it is a high card (>8) assign some expectedtricks it will win
                        # tempProb represents chance that all opponents will have 15-rankOfCard cards in that suit (this way we are guarenteed to win with that card)
                        # Specialprob models the chance that our card is higher rank than all other cards of that suit dealt, AND every player has at least 1 of that suit
                        # In the end, we take the higher of the two probabilities as our answer
                        tempProb = 0
                        specialProb = 1
                        if card.rank == 14:
                            tempProb = self.getSideProb(numCardsInSuit, 1)
                            specialProb = 0
                        elif card.rank == 13:
                            tempProb = self.getSideProb(numCardsInSuit, 2)
                            j = 0
                            k=1
                            for i in range(14,15):
                                if Card(i,suit) in myHand:
                                    specialProb *= 1
                                    k+=1
                                else:
                                    specialProb *= 1 - (self.tricksInRound*(self.numberOfPlayers-1))/(52-self.tricksInRound-j)
                                    j += 1
                                specialProb *= self.getSideProb(numCardsInSuit, k)
                        elif card.rank == 12:
                            tempProb = self.getSideProb(numCardsInSuit, 3)
                            j = 0
                            k=1
                            for i in range(13, 15):
                                if Card(i, suit) in myHand:
                                    specialProb *= 1
                                    k+=1
                                else:
                                    specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                                52 - self.tricksInRound - j)
                                    j += 1
                                specialProb *= self.getSideProb(numCardsInSuit, k)
                        elif card.rank == 11:
                            tempProb = self.getSideProb(numCardsInSuit, 4)
                            j = 0
                            k=1
                            for i in range(12, 15):
                                if Card(i, suit) in myHand:
                                    specialProb *= 1
                                    k+=1
                                else:
                                    specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                                52 - self.tricksInRound - j)
                                    j += 1
                                specialProb *= self.getSideProb(numCardsInSuit, k)
                        elif card.rank == 10:
                            tempProb = self.getSideProb(numCardsInSuit, 5)
                            j = 0
                            k =1
                            for i in range(11, 15):
                                if Card(i, suit) in myHand:
                                    specialProb *= 1
                                    k+=1
                                else:
                                    specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                                52 - self.tricksInRound - j)
                                    j += 1
                                specialProb *= self.getSideProb(numCardsInSuit, k)
                        elif card.rank == 9:
                            j = 0
                            k = 1
                            for i in range(10, 15):
                                if Card(i, suit) in myHand:
                                    specialProb *= 1
                                    k+=1
                                else:
                                    specialProb *= 1 - (self.tricksInRound * (self.numberOfPlayers - 1)) / (
                                                52 - self.tricksInRound - j)
                                    j += 1
                                specialProb *= self.getSideProb(numCardsInSuit, k)
                        if card.rank >= 9:
                            if tempProb > specialProb:
                                # print(card,"Temp prob",tempProb)
                                expectedBid += tempProb
                            else:
                                # print(card, "special prob", specialProb)
                                expectedBid += specialProb
            # Here we deal with trump cards
            sideSuitCuts = {"C":1,
                       "S":1,
                       "H":1,
                       "D":1}
            trumpCardProbs = [[] for x in range(mySuits[self.trumpSuit][0])]
            count = 0
            # Two ways of winning a trump card: beating other trump cards (high card)
            # OR "trumping in" and winning as the only trump
            for trumpCard in sorted(mySuits[self.trumpSuit][2],key=lambda c: c.rank):
                bestProb = 0
                bestProbSuit = ""
                # count each trump card with rank >= 10 as a win
                if trumpCard.rank >= 10:
                    trumpCardProbs[count].append(1)
                else:
                    for suit in ['C', 'S', 'H', 'D']:
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
                    if sum(self.bids) == self.tricksInRound:
                        self.bids[startingPlayer] = round(predBid) + 1
                        bidsInPlayerOrder[p] = round(predBid) + 1
                    elif sum(self.bids)+predBid -self.tricksInRound > 0:
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
            for card in mySuits[self.trumpSuit][2]:
                if card.rank >10:
                    countHighTrump+=1
            if countHighTrump > 0 and round(predBid) == 0:
                predBid += 1
                self.bids[startingPlayer] = round(predBid)
                bidsInPlayerOrder[p] = round(predBid)
                if sum(self.bids)==self.tricksInRound:
                    self.bids[startingPlayer] += 1
                    bidsInPlayerOrder[p] += 1
            if verbose:
                print("PRED BID:", predBid, "Player:", startingPlayer)
                print("ACTUAL BID:", self.bids[startingPlayer], "Player:", startingPlayer, "\n")
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
            cards.append(Card(cardRank,cardSuit))
        return cards

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
        self.discards = []
        self.currentTrick = []
        self.tricksTaken = [0 for p in range(0, self.numberOfPlayers)]
        self.bids = [0 for p in range(0, self.numberOfPlayers)]
        deck = self.GetCardDeck()
        random.shuffle(deck)
        trumpCard = ""
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
            trumpCard = Card(cardRank,cardSuit)
            deck.remove(trumpCard)
        for p in range(0, self.numberOfPlayers):
            if self.enterCards[p]:
                self.playerHands[p] = self.getUserCards(p)
                deck = deepcopy(list(set(deck) - set(self.playerHands[p])))
        for p in range(0, self.numberOfPlayers):
            if not self.enterCards[p]:
                if any(self.enterCards):
                    self.playerHands[p] = deepcopy(deck)
                else:
                    self.playerHands[p] = deepcopy(deck[: self.tricksInRound])
                    deck = deck[self.tricksInRound:]
        if not any(self.enterCards):
            self.flippedOverCard = deck[0]
        else:
            self.flippedOverCard = trumpCard
        self.trumpSuit = self.flippedOverCard.suit
    def GetNextPlayer(self, p):
        """ Return the player to the left of the specified player
        """
        if p == self.numberOfPlayers - 1:
            next = 0
        else:
            next = (p % (self.numberOfPlayers-1)) + 1
        return next
    def notInHand(self,hand,move):
        """
        Helper method for updating particles. Checks if the given move is not in hand.
        However, it treats cards with similar rank as equivalent for the purpose of not
        eliminating so many particles. If a player plays a 5 instead of a 4, it really doesn't matter
        """
        if move.rank <= 6:
            return Card(2,move.suit) not in hand and Card(3,move.suit) not in hand and Card(4,move.suit) not in hand and Card(5,move.suit) not in hand and Card(6,move.suit) not in hand
        elif move.rank <= 9:
            return Card(7,move.suit) not in hand and Card(8,move.suit) not in hand and Card(9,move.suit) not in hand and Card(10,move.suit) not in hand
        elif move.rank <= 11:
            return Card(10,move.suit) not in hand and Card(11,move.suit) not in hand and Card(12,move.suit) not in hand
        elif move.rank <= 13:
            return Card(12, move.suit) not in hand and Card(13, move.suit) not in hand and Card(14,move.suit) not in hand
        elif move.rank <= 14:
            return Card(13, move.suit) not in hand and Card(14, move.suit) not in hand
    def containsSuit(self,hand,suit):
        """
        Checks if the player has any cards in this suit
        """
        for card in hand:
            if card.suit == suit:
                return True
        return False
    def updateOnVoidSuit(self,move):
        """
        First checks if the player is known to be void or not, and if they didn't follow suit.
        If they didn't follow suit and they aren't known to be void, we set them to void in that suit for all players,
        and then update particle weights (if that players' hand in that particle has that suit, it is wrong)
        """
        if move.suit != self.currentTrick[0][1].suit and not self.voids[self.currentTrick[0][1].suit][self.playerToMove]:
            self.voids[self.currentTrick[0][1].suit][self.playerToMove] = True
            for player in self.players:
                if self.isAI[player] and self.useHeuristic[player]:
                    for w, particle in enumerate(self.particles[player]):
                        if player != self.playerToMove:
                            hand = particle[self.playerToMove]
                            if self.containsSuit(hand,self.currentTrick[0][1].suit):
                                self.logWeights[player][w] = -inf
    def checkToRemove(self,move,card):
        """
        Checks if we should remove card from players hand given the move. Are they similar enough to be considered equal
        """
        if card.suit == move.suit:
            if move.rank <= 6:
                if card.rank <= 6:
                    return True
                else:
                    return False
            elif move.rank <= 9:
                if card.rank <= 10 and card.rank > 6:
                    return True
                else:
                    return False
            elif move.rank <= 11:
                if card.rank <= 12 and card.rank >9:
                    return True
                else:
                    return False
            elif move.rank <= 13:
                if card.rank <= 14 and card.rank >11:
                    return True
                else:
                    return False
            elif move.rank <= 14:
                if card.rank == 14 or card.rank == 13:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    def updateOnCardPlayed(self,move):
        """
        Updates particles for each player using them, on a card played. Particles that don't contain
        the card (or similar card as defined above) have weight set to 0
        """
        for player in self.players:
            if self.isAI[player] and self.useHeuristic[player]:
                for w, particle in enumerate(self.particles[player]):
                    if player != self.playerToMove:
                        for play in self.players:
                            if play == self.playerToMove:
                                hand = particle[self.playerToMove]
                                # if not in hand, weight to 0
                                if self.notInHand(hand,move):
                                    self.logWeights[player][w] = -inf
                                else:
                                    # if in hand, either remove if it is exactly equal to move, or go through cards
                                    # and find the similar card and remove that one
                                    if move in hand:
                                        hand.remove(move)
                                    else:
                                        removed = False
                                        i = 0
                                        while not removed:
                                            curCard = hand[i]
                                            if self.checkToRemove(move,curCard):
                                                removed = True
                                                hand.remove(curCard)
                                            i+=1
                            else:
                                # If another player has the card, that is also illegal
                                hand = particle[play]
                                if move in hand:
                                    self.logWeights[player][w] = -inf
                    else:
                        # if the current player played the card, we know our own hand so just remove it from our hand
                        self.particles[player][w][player].remove(move)
    def countZeroLogWeights(self):
        """
        Counts number of particles with -inf logweight (zero probability, not possible)
        """
        counts = [0 for player in self.players]
        for player in self.players:
            for w in self.logWeights[player]:
                if w == -inf:
                    counts[player] += 1
        return counts
    def DoMove(self, move):
        """
        Performs the given move, and updates things as necessary. Particles get updated here as needed
        """
        self.currentTrick.append((self.playerToMove, move))
        self.playerHands[self.playerToMove].remove(move)
        if self.main and self.tricksInRound!=1:
            # If this is the main OhHellState, update particles and regenerate if necessary
            self.updateOnVoidSuit(move)
            self.updateOnCardPlayed(move)
            counts2 = self.countZeroLogWeights()
            for player in self.players:
                if counts2[player] == self.numParticles:
                    self.regenerate(player)
            for player in self.players:
                if self.isAI[player] and self.useHeuristic[player]:
                    self.normalizeWeights(player)
            self.systematic_resample()
        self.playerToMove = self.GetNextPlayer(self.playerToMove)
        # If we are at the end of a trick
        if any(True for (player, card) in self.currentTrick if player == self.playerToMove):
            (leader, leadCard) = self.currentTrick[0]
            suitedPlays = [(player, card.rank) for (player, card) in self.currentTrick if card.suit == leadCard.suit]
            trumpPlays = [(player, card.rank) for (player, card) in self.currentTrick if card.suit == self.trumpSuit]
            # Sort cards based on rules of trick taking games
            sortedPlays = (
                    sorted(suitedPlays, key=lambda pair: pair[1]) +
                    sorted(trumpPlays, key=lambda pair: pair[1])
            )
            # Pick the winner and update stuff
            trickWinner = sortedPlays[-1][0]
            self.tricksTaken[trickWinner] += 1
            self.discards += [card for (player, card) in self.currentTrick]
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
        hand = deepcopy(self.playerHands[self.playerToMove])
        # If leading, play whatever you want
        if self.currentTrick == []:
            return hand
        else:
            (leader, leadCard) = self.currentTrick[0]
            cardsInSuit = [card for card in hand if card.suit == leadCard.suit]
            # follow suit if you can, if not play whatever
            if cardsInSuit != []:
                return cardsInSuit
            else:
                return hand
    def GetResult(self, player):
        """
        Get the game result from the viewpoint of player incorporating scores of other players. Used in ISMCTS
        """
        return (self.GetScore(player) - sum(self.GetScore(p) for p in range(self.numberOfPlayers) if p != player)/(self.numberOfPlayers-1) - self.minRes)/(self.range)
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
    def __repr__(self):
        """
        Return a human-readable representation of the state
        """
        # result = "Round %i" % self.round
        result = "Tricks: %i" % self.tricksInRound
        result += " | Player %i: " % self.playerToMove
        result += ", ".join(str(card) for card in self.playerHands[self.playerToMove])
        result += " | Tricks: %i" % self.tricksTaken[self.playerToMove]
        result += " | Trump: %s" % self.trumpSuit
        result += " | Trick: ["
        result += ",".join(("%i:%s" % (player, card)) for (player, card) in self.currentTrick)
        result += "]"
        return result