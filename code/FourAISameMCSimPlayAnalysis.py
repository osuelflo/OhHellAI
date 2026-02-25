from efficientISMCTS import *

def PlayRound(state,mcIterations,randomRollout,tricksterPlay,data):
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
        moves = state.GetMoves()
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
                            m = CardHelper.str_to_card(card)
                            if CardHelper.has_card(state.GetMoves(),m):
                                bad = False
                            else:
                                print("Illegal move. Try again.")
                        state.DoMove(m)

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
        if len(state.currentTrick) == 1:
            player = state.currentTrick[0][0]
            move = state.currentTrick[0][1]
            suit = CardHelper.get_card_suit(move,isHand=False)
            strmove = CardHelper.to_str(move)
            cardClass = classifyCardRank(move)
            trump = suit == state.trumpSuit
            numSuit = CardHelper.get_suit_num_cards(state.playerHands[player],suit)+1
            tricksLeft = CardHelper.get_num_cards(state.playerHands[player])+1
            tricksToWin = state.GetTricksNeeded(player)
            data["player"].append(player)
            data["card"].append(strmove)
            data["rankCategory"].append(cardClass)
            data["trump"].append(trump)
            data["numSuitInHand"].append(numSuit)
            data["tricksToWin0"].append(tricksToWin)
            data["tricksLeft"].append(tricksLeft)
        # end = time.perf_counter()-start
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

def playFullGame(numPlayers,dealer,main,isAI,useHeuristic,bidStyle,enterCards,mcIterations,randomRollout,tricksterPlay,tricksterBid,data):
    tricks = [x for x in range(10, 0, -1)] + [x for x in range(2, 11)]
    # tricks = [1]
    scoreboard = [0 for i in range(numPlayers)]
    scores = []
    for numTricks in tricks:
        print("DEALER: " + str(dealer))
        state = OhHellState(numPlayers,numTricks,dealer,main,isAI,useHeuristic,bidStyle,enterCards,tricksterBid)
        roundScores = PlayRound(state, mcIterations,randomRollout,tricksterPlay,data)
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

def classifyCardRank(card) :
    rank = CardHelper.get_card_rank(card,isHand=False)
    if rank < 6:
        return "low"
    elif rank < 8:
        return "mediumlow"
    elif rank < 10:
        return "mediumHigh"
    elif rank < 13:
        return "high"
    elif rank < 15:
        return "veryhigh"













allTimeWins = [0,0,0,0]
allTimeScore = [0,0,0,0]
scoreList = []
dealer = 3
count = 1
start = time.perf_counter()
data = {    "player":[],
            "card": [],
            "rankCategory": [],
            "trump": [],
            "numSuitInHand": [],
            "tricksToWin0": [],
            "tricksLeft": []
        }
data2 = {
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
for i in range(100000):
    print("round",i)
    mcIterations = [2500,2500,2500,2500]
    scoreboard,scores = playFullGame(4, dealer, True, [True, True, True, True], [True,True,True,True],["normal", "normal", "normal", "normal"],[False,False,False,False],mcIterations,[True,True,True,True],[False,False,False,False],[False,False,False,False],data)
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
    data2["dealer"].append(dealer)
    if dealer == 3:
        dealer = 0
    else:
        dealer += 1
    data2["winner"].append(winner)
    data2["score1"].append(scoreboard[1])
    data2["score0"].append(scoreboard[0])
    data2["score2"].append(scoreboard[2])
    data2["score3"].append(scoreboard[3])
    data2["round1"].append(scores[0])
    data2["round2"].append(scores[1])
    data2["round3"].append(scores[2])
    data2["round4"].append(scores[3])
    data2["round5"].append(scores[4])
    data2["round6"].append(scores[5])
    data2["round7"].append(scores[6])
    data2["round8"].append(scores[7])
    data2["round9"].append(scores[8])
    data2["round10"].append(scores[9])
    data2["round11"].append(scores[10])
    data2["round12"].append(scores[11])
    data2["round13"].append(scores[12])
    data2["round14"].append(scores[13])
    data2["round15"].append(scores[14])
    data2["round16"].append(scores[15])
    data2["round17"].append(scores[16])
    data2["round18"].append(scores[17])
    data2["round19"].append(scores[18])
    end = time.perf_counter()
    elapsed = end - start
    print(elapsed)
    if elapsed > 1800:
        print("yeet")
        start = time.perf_counter()
        df = pd.DataFrame(data)
        df.to_csv(f"FourAISameMCSimPlayAnalysis{count}HrOne.csv", index=False)
        data = {"player": [],
                "card": [],
                "rankCategory": [],
                "trump": [],
                "numSuitInHand": [],
                "tricksToWin0": [],
                "tricksLeft": []
                }
        df2 = pd.DataFrame(data2)
        df2.to_csv(f"FourAISame2500MCSim{count}HrTwo.csv", index=False)
        data2 = {
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