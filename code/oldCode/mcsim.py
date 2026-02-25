from heuristicISMCTS import *
"""
This is a script to generate the probabilties used in the bidding algorithm
Generates two csvs (one for trump suit and one for off suit) for each combination of numPlayers (2-5) and numTricks (1-10).
Each csv is then a lookup table to find probability each other player has at least X cards in suit
given current player has Y # cards in that suit
"""
for k in range(1,11):
    numRuns = 1000000
    maxNumPlayers = 5
    probabilitiesDict = {"Player 1": [],
                         "Player 2": [],
                         "Player 3": [],
                         "Player 4": [],
                         "Player 5": [],
                         "Trump": []
                         }
    cards = [Card(rank, suit) for rank in range(2, 14 + 1) for suit in ['C', 'D', 'H', 'S']]
    tricksInRound = k
    print("TRICKS ++============",tricksInRound)
    for i in range(numRuns):
        deck = cards.copy()
        random.shuffle(deck)
        probabilitiesDict["Trump"].append([[deck[0]]])
        deck = deck[1:]
        for j in range(1, maxNumPlayers+1):
            probabilitiesDict["Player %i" % j].append([deck[:tricksInRound]])
            clubs = 0
            spades = 0
            hearts = 0
            diamonds = 0
            for card in probabilitiesDict["Player %i" % j][i][0]:
                if card.suit == "S":
                    spades += 1
                if card.suit == "H":
                    hearts += 1
                if card.suit == "D":
                    diamonds += 1
                if card.suit == "C":
                    clubs += 1
            probabilitiesDict["Player %i" % j][i].append(clubs)
            probabilitiesDict["Player %i" % j][i].append(spades)
            probabilitiesDict["Player %i" % j][i].append(hearts)
            probabilitiesDict["Player %i" % j][i].append(diamonds)
            deck = deck[tricksInRound:]
    sideSuitProbTables = {"2P":[],
                          "3P":[],
                           "4P":[],
                          "5P":[]
                         }
    keys = ["2P", "3P", "4P", "5P"]
    for key in keys:
        for mySuits in range(tricksInRound+1):
            temp = []
            for oppSuits in range(tricksInRound+1):
                temp.append([0,0])
            sideSuitProbTables[key].append(temp)
    trumpSuitProbTables = {"2P":[],
                          "3P":[],
                           "4P":[],
                          "5P":[]
                         }
    keys = ["2P", "3P", "4P", "5P"]
    for key in keys:
        for mySuits in range(tricksInRound+1):
            temp = []
            for oppSuits in range(tricksInRound+1):
                temp.append([0,0])
            trumpSuitProbTables[key].append(temp)
    suitKey = {"C":0, "S":1, "H":2, "D":3}
    for hand in range(numRuns):
        myHand = probabilitiesDict["Player %i" % 1][hand]
        myClubs = myHand[1]
        mySpades = myHand[2]
        myHearts = myHand[3]
        myDiamonds = myHand[4]
        myCards = [myClubs, mySpades, myHearts, myDiamonds]
        suits = []
        trumpSuit = probabilitiesDict["Trump"][hand][0][0].suit
        for opp in range(2, maxNumPlayers+1):
            oppHand = probabilitiesDict["Player %i" % opp][hand]
            oppClubs = oppHand[1]
            oppSpades = oppHand[2]
            oppHearts = oppHand[3]
            oppDiamonds = oppHand[4]
            suits.append([oppClubs,oppSpades,oppHearts,oppDiamonds])
        for suit in ["C","S","H","D"]:
            for key in keys:
                for mySuits in range(tricksInRound+1):
                    for oppSuits in range(tricksInRound+1):
                        if key == "2P":
                            if mySuits + oppSuits*1 <= 13 and max(mySuits,oppSuits) <= tricksInRound:
                                if suit == trumpSuit:
                                    if mySuits + oppSuits * 1 <= 12:
                                        if myCards[suitKey[suit]] == mySuits:
                                            trumpSuitProbTables[key][mySuits][oppSuits][1] += 1
                                        if myCards[suitKey[suit]] == mySuits and suits[0][suitKey[suit]] >= oppSuits:
                                            trumpSuitProbTables[key][mySuits][oppSuits][0] += 1
                                else:
                                    if myCards[suitKey[suit]] == mySuits:
                                        sideSuitProbTables[key][mySuits][oppSuits][1] += 1
                                    if myCards[suitKey[suit]] == mySuits and suits[0][suitKey[suit]] >= oppSuits:
                                        sideSuitProbTables[key][mySuits][oppSuits][0] += 1
                        elif key == "3P":
                            if mySuits + oppSuits*2 <= 13 and max(mySuits,oppSuits) <= tricksInRound:
                                if suit == trumpSuit:
                                    if mySuits + oppSuits * 2 <= 12:
                                        if myCards[suitKey[suit]] == mySuits:
                                            trumpSuitProbTables[key][mySuits][oppSuits][1] += 1
                                        if myCards[suitKey[suit]] == mySuits and suits[0][suitKey[suit]] >= oppSuits and suits[1][suitKey[suit]] >= oppSuits:
                                            trumpSuitProbTables[key][mySuits][oppSuits][0] += 1
                                else:
                                    if myCards[suitKey[suit]] == mySuits:
                                        sideSuitProbTables[key][mySuits][oppSuits][1] += 1
                                    if myCards[suitKey[suit]] == mySuits and suits[0][suitKey[suit]] >= oppSuits and suits[1][suitKey[suit]] >= oppSuits:
                                        sideSuitProbTables[key][mySuits][oppSuits][0] += 1
                        elif key == "4P":
                            if mySuits + oppSuits*3 <= 13 and max(mySuits,oppSuits) <= tricksInRound:
                                if suit == trumpSuit:
                                    if mySuits + oppSuits * 3 <= 12:
                                        if myCards[suitKey[suit]] == mySuits:
                                            trumpSuitProbTables[key][mySuits][oppSuits][1] += 1
                                        if myCards[suitKey[suit]] == mySuits and suits[0][suitKey[suit]] >= oppSuits and \
                                                suits[1][suitKey[suit]] >= oppSuits and suits[2][suitKey[suit]] >= oppSuits:
                                            trumpSuitProbTables[key][mySuits][oppSuits][0] += 1
                                else:
                                    if myCards[suitKey[suit]] == mySuits:
                                        sideSuitProbTables[key][mySuits][oppSuits][1] += 1
                                    if myCards[suitKey[suit]] == mySuits and suits[0][suitKey[suit]] >= oppSuits and suits[1][suitKey[suit]] >= oppSuits and suits[2][suitKey[suit]] >= oppSuits:
                                        sideSuitProbTables[key][mySuits][oppSuits][0] += 1
                                        # print(key, sideSuitProbTables[key][mySuits][oppSuits], "ADDED ONE")
                        elif key == "5P":
                            if mySuits + oppSuits*4 <= 13 and max(mySuits,oppSuits) <= tricksInRound:
                                if suit == trumpSuit:
                                    if mySuits + oppSuits*4 <= 12:
                                        if myCards[suitKey[suit]] == mySuits:
                                            trumpSuitProbTables[key][mySuits][oppSuits][1] += 1
                                        if myCards[suitKey[suit]] == mySuits and suits[0][suitKey[suit]] >= oppSuits and \
                                                suits[1][suitKey[suit]] >= oppSuits and suits[2][suitKey[suit]] >= oppSuits and \
                                                suits[3][suitKey[suit]] >= oppSuits:
                                            trumpSuitProbTables[key][mySuits][oppSuits][0] += 1
                                else:
                                    if myCards[suitKey[suit]] == mySuits:
                                        sideSuitProbTables[key][mySuits][oppSuits][1] += 1
                                    if myCards[suitKey[suit]] == mySuits and suits[0][suitKey[suit]] >= oppSuits and suits[1][suitKey[suit]] >= oppSuits and suits[2][suitKey[suit]] >= oppSuits and suits[3][suitKey[suit]] >= oppSuits:
                                        sideSuitProbTables[key][mySuits][oppSuits][0] += 1
    import pandas as pd

    dfs = {}  # store separate DataFrames

    for game_key, suit_list in sideSuitProbTables.items():
        rows = []
        for mysuit, oppsuit in enumerate(suit_list):
            for oppsuitnum, metrics in enumerate(oppsuit):
                if not isinstance(metrics, list) or len(metrics) != 2:
                    continue
                rows.append({
                    "numMySuit": mysuit,
                    "numAtLeastOppSuit": oppsuitnum,
                    "numTargetHit": metrics[0],
                    "numTotal": metrics[1]
                })
        df = pd.DataFrame(rows)
        df = df[df["numTotal"] > 0]
        df["probability"] = df["numTargetHit"] / df["numTotal"]
        dfs[game_key] = df
        dfs[game_key].to_csv(f"{tricksInRound}Tricks{game_key}SideSuit.csv", index=False)

    for game_key, suit_list in trumpSuitProbTables.items():
        rows = []
        for mysuit, oppsuit in enumerate(suit_list):
            for oppsuitnum, metrics in enumerate(oppsuit):
                if not isinstance(metrics, list) or len(metrics) != 2:
                    continue

                rows.append({
                    "numMySuit": mysuit,
                    "numAtLeastOppSuit": oppsuitnum,
                    "numTargetHit": metrics[0],
                    "numTotal": metrics[1]
                })
        df = pd.DataFrame(rows)
        df = df[df["numTotal"] > 0]
        df["probability"] =  df["numTargetHit"] / df["numTotal"]
        dfs[game_key] = df
        dfs[game_key].to_csv(f"{tricksInRound}Tricks{game_key}TrumpSuit.csv", index=False)
