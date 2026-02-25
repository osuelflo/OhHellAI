


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
from heuristicISMCTS import *
from playGame import *
import pandas as pd
verbose = False

def PlayManyRounds(data,numRounds,numPlayers,numTricks,numParticles,lamb,isAI,useHeuristic,mcIterations,selfish,aggs):

    for game in range(numRounds):
        dealer = 3
        # PLAY ROUND
        state = OhHellState(numPlayers, numTricks, numParticles, lamb, dealer, True, isAI, useHeuristic, aggs)
        tricksToWin = state.bids
        while (state.GetMoves() != []):
            player = state.playerToMove
            iterations = mcIterations[player]
            useScore = selfish[player]
            if len(state.GetMoves()) == 1:
                state.DoMove(state.GetMoves()[0])
            else:
                if iterations == 0:
                    m = random.choice(state.GetMoves())
                    state.DoMove(m)
                else:
                    if len(state.currentTrick) == 0:
                        data["tricksLeft"].append(len(state.playerHands[player]))
                        data["tricksToWin0"].append(state.bids[0]-state.tricksTaken[0])
                        data["tricksToWin1"].append(state.bids[1] - state.tricksTaken[1])
                        data["tricksToWin2"].append(state.bids[2] - state.tricksTaken[2])
                        data["tricksToWin3"].append(state.bids[3] - state.tricksTaken[3])
                    m = ISMCTS(rootstate=state, itermax=iterations, verbose=False, selfish=useScore)
                    if len(state.currentTrick) == 0:
                        data["card"].append(str(m))
                        curSuit = m.suit
                        data["trump"].append(curSuit == state.trumpSuit)
                        data["numSuitInHand"].append(sum(card.suit == curSuit for card in state.playerHands[player]))
                        data["rankCategory"].append(classifyCardRank(m))
                        # print(m)
                    state.DoMove(m)



def classifyCardRank(card) :
    if card.rank < 6:
        return "low"
    elif card.rank < 8:
        return "mediumlow"
    elif card.rank < 10:
        return "mediumHigh"
    elif card.rank < 13:
        return "high"
    elif card.rank < 15:
        return "veryhigh"

if __name__ == "__main__":
    import time
    start = time.perf_counter()
    count = 0
    for i in range(10000):
        numRounds = 100
        numPlayers = 4
        numTricks = 10
        numParticles = 1000
        lamb=1.3
        isAI = [True,True,True,True]
        useHeuristic = [False,False,False,False]
        mcIterations = [1000,0,0,0]
        selfish = [True,True,True,True]
        aggs = ["normal","normal","normal","normal"]
        data = {
            "card": [],
            "rankCategory": [],
            "trump": [],
            "numSuitInHand": [],
            "tricksToWin0": [],
            "tricksToWin1": [],
            "tricksToWin2": [],
            "tricksToWin3": [],
            "tricksLeft": []
        }
        PlayManyRounds(data, numRounds, numPlayers, numTricks, numParticles, lamb, isAI, useHeuristic, mcIterations,
                       selfish, aggs)
        end = time.perf_counter()
        elapsed = end - start
        if elapsed > 3600:
            start = time.perf_counter()
            df = pd.DataFrame(data)
            df.to_csv(f"playsOutput{count}HrEight.csv", index=False)
            count += 1
            data = {
                "card": [],
                "rankCategory": [],
                "trump": [],
                "numSuitInHand": [],
                "tricksToWin0": [],
                "tricksToWin1": [],
                "tricksToWin2": [],
                "tricksToWin3": [],
                "tricksLeft": []
            }