from efficientISMCTS import *
for i in range(1):
    mcIterations = [0,1000,1000,1000]
    dealer = 3
    numPlayers = 4
    numTricks = 10
    main = True
    isAI = [False, True, True, True]
    randomRollout = [False, True, True, True]
    useHeuristic = [False, True, True, True]
    bidStyle = ["normal", "normal", "normal", "normal"]
    enterCards = [False,False,False,False]
    trickster = [False,False,False,False]
    playFullGame(numPlayers,dealer,main,isAI,useHeuristic,bidStyle,enterCards,mcIterations,randomRollout,trickster,trickster)

#
# with open("bidTestingNotes.txt", "w") as f:
#     f.write("num_tricks,num_players,player,hand,flipped_card,pred_bid,actual_bid\n")
# for i in range(2,6):
#     for j in range(10,1,-1):
#         with open("bidTestingNotes.txt", "a") as f:
#             f.write("\n")
#             f.write(str(j))
#             f.write(",")
#             f.write(str(i))
#             f.write(",")
#         state = OhHellState(4, j, 1000, 1.3, 3, True, [True, True, True, True], [False, False, False, False],
#                             ["normal", "normal", "normal", "normal"], [False, False, False, False])
#
#         for n in range(25):
#             state.Deal()
#             bidsInPlayerOrder = [0 for j in range(0, state.numberOfPlayers)]
#             startingPlayer = state.dealer
#             for p in state.players:
#                 with open("bidTestingNotes.txt", "a") as f:
#                     f.write("\n")
#                     f.write(str(j))
#                     f.write(",")
#                     f.write(str(i))
#                     f.write(",")
#                 startingPlayer = state.GetNextPlayer(startingPlayer)
#                 state.Bid(bidsInPlayerOrder,p,startingPlayer)

# with open("bidTestingNotes.txt", "r") as f:
#     lines = f.readlines()
#
# filtered = [line for line in lines if len(line.strip()) < 100]
#
# with open("bidTestingData.csv", "w") as f:
#     f.writelines(filtered)
# totalBids = [0,0,0,0]
# state = OhHellState(4, 10, 3, True, [True, True, True, True], [True, False, False, False],
#                                 ["normal", "normal", "normal", "normal"], [False, False, False, False])
# print(state.bids)
# state.printHand(0)
# random.seed(1003)
# for i in range(1000):
#     # for i in range(5):
#     #     # state.printHand(i)
#     #     print(state.probTables[0][i])
#
#
#     state.randomDeal(0,state.probTables)
#     state.bids = [0,0,0,0]
#     bidsInPlayerOrder = [0 for j in range(0, state.numberOfPlayers)]
#     startingPlayer = state.dealer
#     for p in state.players:
#         startingPlayer = state.GetNextPlayer(startingPlayer)
#         state.Bid(bidsInPlayerOrder, p,startingPlayer)
#         totalBids[p] += state.bids[p]
#
#
# for i in totalBids:
#     print(i/1000)
# print(state.probTables[0][0])
# print(state.probTables[0][1])
# print(state.probTables[0][2])
# print(state.probTables[0][3])
# print(state.probTables[0][4])
# random.seed(101)
#
# state = OhHellState(4, 10, 3, True, [True, True, True, True], [True, False, False, False],
#                                 ["normal", "normal", "normal", "normal"], [False, False, False, False])
#
# # st = state.CloneAndRandomize(0)
# # st.reMakeProbTable(0,1,state.probTables)
# # state.printProbTables(0,1)
# for i in np.arange(0.001,10,0.001):
#     for j in np.arange(0.001,1,0.001):
#         if j + state.probChange(i,j) > 1:
#             print(i,j,state.probChange(i,j))


# import random
#

# nums = nums + [0.2]
# print(sum(nums))
# change = 0.8-random.random()
# for i in range(len(nums)-1):
#     old = nums[4]
#     new = nums[4] + change
#     nums[i] *= (1-new)/(1-old)
# print(sum(nums))


import numpy as np

# def redistribute(p, k, p_k_new):
#     p = np.array(p, dtype=float)
#     p_k_old = p[k]
#     change = p_k_new-p_k_old
#     sum_other = sum([p[i] for i in range(len(p)) if i != k])
#     print(sum_other)
#     for i in range(len(p)):
#         if i == k:
#             p[i] = p_k_new
#         else:
#             p[i] = p[i]-change*p[i]/sum_other
#     return p
#
# # p = np.array([0.1, 0.2, 0.3, 0.4])
# nums = [random.random() for _ in range(4)]
# total = sum(nums)
#
# p = [x / total for x in nums]
# for i in [0,1,3]:
#     for j in [0,1,3]:
#         print(p[i]/p[j])
# print("-----------------------")
# print(p)
# p_new = redistribute(p, k=2, p_k_new=0.5)
# print(p_new, p_new.sum())
# for i in [0,1,3]:
#     for j in [0,1,3]:
#         print(p[i]/p[j])