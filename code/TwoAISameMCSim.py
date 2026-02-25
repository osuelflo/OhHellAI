from efficientISMCTS import *
allTimeWins = [0,0]
allTimeScore = [0,0]
scoreList = []
dealer = 1
data = {
    "winner": [],
    "score1": [],
    "score2": [],
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
count = 1
start = time.perf_counter()
for i in range(100000):

    mcIterations = [2500,2500]
    scoreboard,scores = playFullGame(2, dealer, True, [True, True], [True,True],["normal", "normal"],[False,False],mcIterations,[True,True],[False,False],[False,False])
    # mcIterations = [1000,250,100,0]
    # scores = playFullGame(4, dealer, True, [True, True, True, False], [True,True,True,False],["normal", "normal", "normal", "normal"],[False,False,False,False],mcIterations,[True,True,True,False])
    winner = 0
    scoreList.append(scoreboard)
    highScore = scoreboard[0]
    for i in range(2):
        allTimeScore[i] += scoreboard[i]
        if scoreboard[i] >= highScore:
            winner = i
            highScore = scoreboard[i]
    allTimeWins[winner] += 1
    data["dealer"].append(dealer)
    if dealer == 1:
        dealer = 0
    else:
        dealer += 1
    data["winner"].append(winner)
    data["score1"].append(scoreboard[1])
    data["score0"].append(scoreboard[0])
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
    if elapsed > 1800:
        print("yeet")

        start = time.perf_counter()
        df = pd.DataFrame(data)
        df.to_csv(f"TwoAISameMCSim{count}HrOne.csv", index=False)
        data = {
            "winner": [],
            "score1": [],
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