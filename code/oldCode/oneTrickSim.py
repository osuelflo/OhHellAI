from heuristicISMCTS import *
"""
Script to find probability that, given the number of players in the game 
and the spot in the order in which the current player is bidding, what is the chance
they win the trick given their card. This is only for 1 trick rounds.
"""
k = 1
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
    probabilitiesDict["Trump"].append(deck[0])
    deck = deck[1:]
    for j in range(1, maxNumPlayers+1):
        probabilitiesDict["Player %i" % j].append(deck[:tricksInRound][0])
        deck = deck[tricksInRound:]

sideProbTable = []
for numPlayers in range(6):
    playerList = []
    for order in range(numPlayers):
        playerList.append([])
        for rank in range(15):
            playerList[order].append([0,0])
    sideProbTable.append(playerList)

trumpProbTable = []
for numPlayers in range(6):
    playerList = []
    for order in range(numPlayers):
        playerList.append([])
        for rank in range(15):
            playerList[order].append([0, 0])
    trumpProbTable.append(playerList)

for hand in range(numRuns):
    cards = []
    cards.append(probabilitiesDict["Player 1"][hand])
    leadingSuit = cards[0].suit
    trumpSuit = probabilitiesDict["Trump"][hand].suit
    winningPlayer = 0
    winningPlayerCard = cards[0]
    cards.append(probabilitiesDict["Player 2"][hand])
    if cards[1].suit == winningPlayerCard.suit and cards[1].rank > winningPlayerCard.rank:
        winningPlayer = 1
        winningPlayerCard = cards[1]
    elif cards[1].suit == trumpSuit and winningPlayerCard.suit != trumpSuit:
        winningPlayer = 1
        winningPlayerCard = cards[1]
    for player in range(2):
        if player == winningPlayer:
            if cards[player].suit == trumpSuit:
                trumpProbTable[2][winningPlayer][winningPlayerCard.rank][0] += 1
                trumpProbTable[2][winningPlayer][winningPlayerCard.rank][1] += 1
            else:
                sideProbTable[2][winningPlayer][winningPlayerCard.rank][0] += 1
                sideProbTable[2][winningPlayer][winningPlayerCard.rank][1] += 1
        else:
            if cards[player].suit == trumpSuit:
                trumpProbTable[2][player][cards[player].rank][1] += 1
            else:
                sideProbTable[2][player][cards[player].rank][1] += 1
    cards.append(probabilitiesDict["Player 3"][hand])
    if cards[2].suit == winningPlayerCard.suit and cards[2].rank > winningPlayerCard.rank:
        winningPlayer = 2
        winningPlayerCard = cards[2]
    elif cards[2].suit == trumpSuit and winningPlayerCard.suit != trumpSuit:
        winningPlayer = 2
        winningPlayerCard = cards[2]
    for player in range(3):
        if player == winningPlayer:
            if cards[player].suit == trumpSuit:
                trumpProbTable[3][winningPlayer][winningPlayerCard.rank][0] += 1
                trumpProbTable[3][winningPlayer][winningPlayerCard.rank][1] += 1
            else:
                sideProbTable[3][winningPlayer][winningPlayerCard.rank][0] += 1
                sideProbTable[3][winningPlayer][winningPlayerCard.rank][1] += 1
        else:
            if cards[player].suit == trumpSuit:
                trumpProbTable[3][player][cards[player].rank][1] += 1
            else:
                sideProbTable[3][player][cards[player].rank][1] += 1
    cards.append(probabilitiesDict["Player 4"][hand])
    if cards[3].suit == winningPlayerCard.suit and cards[3].rank > winningPlayerCard.rank:
        winningPlayer = 3
        winningPlayerCard = cards[3]
    elif cards[3].suit == trumpSuit and winningPlayerCard.suit != trumpSuit:
        winningPlayer = 3
        winningPlayerCard = cards[3]
    for player in range(4):
        if player == winningPlayer:
            if cards[player].suit == trumpSuit:
                trumpProbTable[4][winningPlayer][winningPlayerCard.rank][0] += 1
                trumpProbTable[4][winningPlayer][winningPlayerCard.rank][1] += 1
            else:
                sideProbTable[4][winningPlayer][winningPlayerCard.rank][0] += 1
                sideProbTable[4][winningPlayer][winningPlayerCard.rank][1] += 1
        else:
            if cards[player].suit == trumpSuit:
                trumpProbTable[4][player][cards[player].rank][1] += 1
            else:
                sideProbTable[4][player][cards[player].rank][1] += 1
    cards.append(probabilitiesDict["Player 5"][hand])
    if cards[4].suit == winningPlayerCard.suit and cards[4].rank > winningPlayerCard.rank:
        winningPlayer = 4
        winningPlayerCard = cards[4]
    elif cards[4].suit == trumpSuit and winningPlayerCard.suit != trumpSuit:
        winningPlayer = 4
        winningPlayerCard = cards[4]
    for player in range(5):
        if player == winningPlayer:
            if cards[player].suit == trumpSuit:
                trumpProbTable[5][winningPlayer][winningPlayerCard.rank][0] += 1
                trumpProbTable[5][winningPlayer][winningPlayerCard.rank][1] += 1
            else:
                sideProbTable[5][winningPlayer][winningPlayerCard.rank][0] += 1
                sideProbTable[5][winningPlayer][winningPlayerCard.rank][1] += 1
        else:
            if cards[player].suit == trumpSuit:
                trumpProbTable[5][player][cards[player].rank][1] += 1
            else:
                sideProbTable[5][player][cards[player].rank][1] += 1
import pandas as pd


def prob_table_to_df(prob_table, table_type):
    """
    prob_table: list[numPlayers][playerOrder][rank] = [wins, outcomes]
    table_type: "side" or "trump"
    """
    rows = []

    for num_players in range(len(prob_table)):
        player_list = prob_table[num_players]

        for player_order in range(len(player_list)):
            rank_list = player_list[player_order]

            for rank in range(len(rank_list)):
                wins, outcomes = rank_list[rank]

                rows.append({
                    "num_players": num_players,
                    "player_order": player_order,
                    "rank": rank,
                    "wins": wins,
                    "outcomes": outcomes,
                    "probability": wins / outcomes if outcomes > 0 else 0.0,
                    "table_type": table_type
                })
    df = pd.DataFrame(rows)
    df= df[df["outcomes"] > 0]
    return df

# Convert both large structures
df_side = prob_table_to_df(sideProbTable, "side")
df_trump = prob_table_to_df(trumpProbTable, "trump")
df_side.to_csv("sideOneTrickProbs.csv", index=False)
df_trump.to_csv("trumpOneTrickProbs.csv", index=False)
