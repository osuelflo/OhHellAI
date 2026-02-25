from heuristicISMCTS import *
verbose = True
verboseHuman = True
# This is a very simple Python 2.7 implementation of the Information Set Monte Carlo Tree Search algorithm.
# The function ISMCTS(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a
# state.GetRandomMove() or state.DoRandomRollout() function.
#
#
# Written by Peter Cowling, Edward Powley, Daniel Whitehouse (University of York, UK) September 2012 - August 2013.
#
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
#
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai
# Also read the article accompanying this code at ***URL HERE***



class Node:
    """
    A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
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
    st = ""
    for node in rootnode.childNodes:
        m = node.move
        vis = node.visits
        st += str(m)+ ": " +str(vis)+"| "
    print(st)
    return max(rootnode.childNodes, key=lambda c: c.visits).move  # return the move that was most visited
def toCard(string):
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
    return Card(rank,suit)
def PlayRound(state,mcIterations,selfish):
    """
    Plays one round of Oh Hell.
    mcIterations is a list of the number of MC iterations to use for each player.
    selfish is a list of booleans, with true meaning we should use the "selfish" scoring for ISMCTS,
    and false meaning we incorporate other players scores into our score for ISMCTS.
    """
    # While moves are still left in the game
    while (state.GetMoves() != []):
        player = state.playerToMove
        iterations = mcIterations[player]
        useScore = selfish[player]
        # If we only have one valid move, play it as that is our only choice
        if len(state.GetMoves()) == 1:
            m=state.GetMoves()[0]
            if verboseHuman:
                print(
                "Player " + str(state.playerToMove) + " Plays " + str(m))
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
                        move = toCard(card)
                        if move in state.GetMoves():
                            bad = False
                        else:
                            print("Illegal move. Try again.")
                    state.DoMove(move)
                else:
                    m = random.choice(state.GetMoves())
                    if verboseHuman:
                        print(
                        "Player " + str(state.playerToMove) + " Plays " + str(m))
                    state.DoMove(m)

            else:
                if iterations == 1000:
                    if verboseHuman and state.enterCards[state.playerToMove]:
                        state.printHand(state.playerToMove)
                    if verbose:
                        state.printHand(state.playerToMove)
                m = ISMCTS(rootstate=state, itermax=iterations,verbose=False,selfish = useScore)
                if verboseHuman:
                    print(
                    "Player " + str(state.playerToMove) + " Plays "+str(m))
                state.DoMove(m)
    for p in range(0, state.numberOfPlayers):
        score = state.GetActualScore(p)
        print(
        "Player " + str(p) + " scores: " + str(score) + "\n")

def playFullGame(numPlayers,numParticles,lamb,dealer,main,isAI,useHeuristic,bidStyle,enterCards,mcIterations,selfish):
    tricks = [x for x in range(3, 0, -1)] + [x for x in range(2, 11)]
    scoreboard = [0, 0, 0, 0]
    for numTricks in tricks:
        print("DEALER: " + str(dealer))
        state = OhHellState(numPlayers,numTricks,numParticles,lamb,dealer,main,isAI,useHeuristic,bidStyle,enterCards)
        PlayRound(state, mcIterations, selfish)
        dealer = state.GetNextPlayer(dealer)
        for p in state.players:
            scoreboard[p] += state.GetActualScore(p)
        print("SCOREBOARD:")
        print("Player One:", scoreboard[0])
        print("Player Two:", scoreboard[1])
        print("Player Three:", scoreboard[2])
        print("Player Four:", scoreboard[3], "\n")

if __name__ == "__main__":
    random.seed(1000)
    mcIterations = [0, 0, 0, 1000]
    selfish = [True,True,True,True]
    playFullGame(4,1000, 1.3, 3, True, [False,False,False,True], [False, False, False, True],["normal", "normal", "normal", "normal"],[False,False,False,True],mcIterations,selfish)


