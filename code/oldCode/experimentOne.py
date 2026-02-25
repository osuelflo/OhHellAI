# Tests to run
#
# Do this for both particle and non particle heuristics
# Various AIs against 3 randoms
# Various AIs against themselves
#
# Metrics to get
# For each game
#   For each player
#       Game Number, Score, Tricks Won, Who Lead in Beginning, Number of MCTS iterations, Lambda, Number of particles, Heuristic or Not, Number of players, Number of tricks
#
#
#
#
# -------------

import numpy as np
from math import *
import random, sys
from copy import deepcopy
from heuristicISMCTS import *
from noheuristicISMCTS import OhHellNoHeurState
import numpy as np
import pandas as pd
verbose = False




class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
    """

    def __init__(self, selfish,move=None, parent=None, playerJustMoved=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.avails = 1
        self.playerJustMoved = playerJustMoved  # the only part of the state that the Node needs later
        self.selfish = selfish

    def GetUntriedMoves(self, legalMoves):
        """ Return the elements of legalMoves for which this node does not have children.
        """

        # Find all moves for which this node *does* have children
        triedMoves = [child.move for child in self.childNodes]

        # Return all moves that are legal but have not been tried yet
        return [move for move in legalMoves if move not in triedMoves]

    def UCBSelectChild(self, legalMoves, exploration=0.7):
        """ Use the UCB1 formula to select a child node, filtered by the given list of legal moves.
            exploration is a constant balancing between exploitation and exploration, with default value 0.7 (approximately sqrt(2) / 2)
        """

        # Filter the list of children by the list of legal moves
        legalChildren = [child for child in self.childNodes if child.move in legalMoves]

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
        n = Node(self.selfish,move=m, parent=self, playerJustMoved=p)
        self.childNodes.append(n)
        return n

    def Update(self, terminalState):
        """ Update this node - increment the visit count by one, and increase the win count by the result of terminalState for self.playerJustMoved.
        """
        self.visits += 1
        if self.playerJustMoved is not None:
            if self.selfish:
                self.wins += terminalState.GetScore(self.playerJustMoved)
            else:
                self.wins += terminalState.GetResult(self.playerJustMoved)

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


def ISMCTS(rootstate, itermax, selfish,verbose=False):
    """ Conduct an ISMCTS search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
    """

    rootnode = Node(selfish)

    for i in range(itermax):
        node = rootnode

        # Determinize
        state = rootstate.CloneAndRandomize(rootstate.playerToMove)

        # Select
        while state.GetMoves() != [] and node.GetUntriedMoves(
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
        while state.GetMoves() != []:  # while state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None:  # backpropagate from the expanded node and work back to the root node
            node.Update(state)
            node = node.parentNode

    # Output some information about the tree - can be omitted
    # if (verbose): print(
    # rootnode.TreeToString(0))
    # else:
    #     print(
    #     rootnode.ChildrenToString())

    return max(rootnode.childNodes, key=lambda c: c.visits).move  # return the move that was most visited

def toCard(string):
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
    return Card(rank,suit)


def PlayManyRounds(data,numRounds,numPlayers,numTricks,numParticles,lamb,isAI,useHeuristic,mcIterations,selfish):
    for game in range(numRounds):
        random.seed(game)
        dealer = random.randint(0,numPlayers-1)
        scores, tricksWon,bids = PlayRound(numPlayers,numTricks,numParticles,lamb,dealer,isAI,useHeuristic,mcIterations,selfish)
        for i in range(numPlayers):
            data["gameID"].append(game)
            data["player"].append(i)
            data["numTricks"].append(numTricks)
            data["numParticles"].append(numParticles)
            data["lambda"].append(lamb)
            data["useHeuristic"].append(useHeuristic[i])
            data["mcIterations"].append(mcIterations[i])
            data["selfish"].append(selfish[i])
            data["score"].append(scores[i])
            data["tricksWon"].append(tricksWon[i])
            data["bid"].append(bids[i])
            data["dealer"].append(dealer)


def PlayRound(numPlayers,numTricks,numParticles,lamb,dealer,isAI,useHeuristic,mcIterations,selfish):
    # if useHeuristic:
    state = OhHellHeurState(numPlayers, numTricks, numParticles, lamb, dealer, True, isAI,useHeuristic)
    # else:
    #     state = OhHellNoHeurState(numPlayers, numTricks, numParticles, lamb, dealer, True, isAI)
    while (state.GetMoves() != []):
        player = state.playerToMove
        iterations = mcIterations[player]
        useScore = selfish[player]
        if len(state.GetMoves()) == 1:
            state.DoMove(state.GetMoves()[0])
        else:
            if iterations == 0:
                if not state.isAI[player]:
                    state.printHand(player)
                    bad = True
                    while bad:
                        card = input("Type the card you want to play ")
                        move = toCard(card)
                        if move in state.GetMoves():
                            bad = False
                        else:
                            print("Illegal move. Try again.")
                    state.DoMove(move)
                else:
                    m = random.choice(state.GetMoves())
                    # print(m)
                    state.DoMove(m)
            else:
                m = ISMCTS(rootstate=state, itermax=iterations,verbose=False,selfish = useScore)
                # print(m)
                state.DoMove(m)
    scores = []
    tricksWons = []
    bids = []
    for p in range(0, state.numberOfPlayers):
        score = state.GetScore(p)
        scores.append(score)
        tricksWon = state.tricksTaken[p]
        tricksWons.append(tricksWon)
        bid = state.bids[p]
        bids.append(bid)
    return scores, tricksWons, bids


if __name__ == "__main__":
    import time
    # random.seed(21)
    # state = OhHellHeurState(4,10,300,1.3,3,True,[True,False,True,True])
    # # print("bids",state.bids)
    # # for p in range(0, state.numberOfPlayers):
    # #     state.printHand(p)
    # mcIterations = [1000, 0, 100, 0]
    # selfish = [True, True, True, True]
    # PlayRound(state,mcIterations,selfish)


    # 9 experiments can be done
    # One AI everyone playing randomly, AI varying number of MC iterations and number of particles and use of heuristic
    # One AI everyone playing randomly, AI varying choice of lambda, number of particles
    # One AI everyone playing randomly, varying use of heuristic and passive/aggressive
    # All AI, varying heuristic and having each AI have different number of mc sims
    # All AI, same mc sim but varying number of particles (LOW) and scoring metrix
    # All AI, same mc sim but varying number of particles (HIGH) and scoring metrix
    # All AI, same mc sim but one aggressive and one passive no heuristic
    # All AI, same mc sim but one aggressive and one passive heuristic

    # Plan:
    # initialize data, and variables that wont change
    # outer-most loop: Changing whatever variable we want to look at MAKE THIS INNER LOOP
    # Then loops for players in game (2,3,4) and number of tricks (all)
    # check if elapsed time greater than an hour, if so
    #   save data to csv
    #   reset data
    #   reset start to be current time
    #
    # ---------------------



    start = time.perf_counter()

    # code you want to time
    particles = [10,50,100,500,1000,2000]
    mcsims = [10,50,100,200,500,1000]
    heuristic = [True,False]
    data = {
        "gameID": [],
        "player": [],
        "numTricks": [],
        "numParticles": [],
        "lambda": [],
        "useHeuristic": [],
        "mcIterations": [],
        "selfish": [],
        "score": [],
        "tricksWon": [],
        "bid": [],
        "dealer": []
    }
    lamb = 1.3
    count = 0
    for numTricks in range(10,0,-1):
        for numPlayers in range(4,1,-1):
            isAI = [True for i in range(numPlayers)]
            selfish = [True for i in range(numPlayers)]
            for mc in mcsims:
                mcIterations = [0 for i in range(numPlayers)]
                mcIterations[0] = mc
                for part in particles:
                    numParticles = part
                    for heur in heuristic:
                        useHeuristic = [False for i in range(numPlayers)]
                        useHeuristic[0] = heur
                        PlayManyRounds(data,100,numPlayers,numTricks,numParticles,lamb,isAI,useHeuristic,mcIterations,selfish)
                        end = time.perf_counter()
                        elapsed = end - start
                        # print(elapsed)
                        if elapsed > 3600:
                            start = time.perf_counter()
                            df = pd.DataFrame(data)
                            df.to_csv(f"output{count}HrOne.csv", index=False)
                            count += 1
                            data = {
                                "gameID": [],
                                "player": [],
                                "numTricks": [],
                                "numParticles": [],
                                "lambda": [],
                                "useHeuristic": [],
                                "mcIterations": [],
                                "selfish": [],
                                "score": [],
                                "tricksWon": [],
                                "bid": [],
                                "dealer": []
                            }
    # minutes, seconds = divmod(elapsed, 60)
    # print(f"Elapsed time: {int(minutes)} minutes {seconds:.2f} seconds")

# Two minutes for 10 trick, 4 person game with 1000 particles and 1000 sims for each player, lamb = 1.3
# Two minutes for 10 trick, 4 person game with 1000 particles ONLY ONE PERSON USING PARTICLES and 1000 sims for each player, lamb = 1.3
# 48 seconds for 10 trick, 4 person game with NO HEURISTIC and 1000 sims for each player, lamb = 1.3
