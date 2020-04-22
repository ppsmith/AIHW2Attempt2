from Type1ModelFree import Type1ModelFree
from Type2ModelFree import Type2ModelFree
from Type3ModelFree import Type3ModelFree

from CloseToPin import CloseToPin
from InTheHole import InTheHole
from LeftOfThePin import LeftOfThePin
from SameLevelAsPin import SameLevelAsPin
from OverTheGreen import OverTheGreen
from Fairway import Fairway1
from Ravine import Ravine1
from PastPin import *
from LeftPin import *
from AtPin import *
from Chip import *
from Pitch import *
from Putt import *
import numpy as np
import random
import sys
import pylab as pl
import networkx as nx

#Type 1 = Ravine and Fairway
#Type 2 = Close, Same, Left, In ("on the green")
#Type 3 = Over the Green

# Fairway = Type1ModelFree("Fairway")
# Ravine = Type1ModelFree("Ravine")
# Close = Type2ModelFree("Close")
# Same = Type2ModelFree("Same")
# Left = Type2ModelFree("Left")
# In = Type2ModelFree("In")
# Over = Type3ModelFree("Over")
#
# #Find local location of input
# #data = open(input('Please give file path:'), 'r')
#
# data = open('C:\\Users\\ppsmith\\Desktop\\learning.txt')
#
# #read each line of data
# for entry in data:
#     #make sure that data hasn't ended
#     if entry != '\n':
#         #Each line goes start location, action, end location, and probability of ending in end state
#         #Each aspect is separated by a /, so find positions of /
#         sub = entry
#         startlocation = "invalid"
#         action = "invalid"
#         endlocation = "invalid"
#         prob = "0"
#         current_index = 0
#         num_slash = 0
#         #break up each line to find the start location, the action, the end location, and the probability
#         while (current_index != -1):
#             current_index = sub.find("/")
#             print(current_index)
#             num_slash = num_slash + 1
#             if (num_slash == 1): #start location
#                 startlocation = sub[0:current_index]
#                 sub = sub[current_index + 1:len(sub)] #create substring starting from action
#             elif (num_slash == 2): #action
#                 action = sub[0:current_index] #create substring starting from end location
#                 sub = sub[current_index + 1:len(sub)]
#             elif (num_slash == 3): #end location and probability
#                 endlocation = sub[0:current_index]
#                 prob = sub[current_index + 2:len(sub)]
#                 sub = sub[current_index + 1:len(sub)]
#             else:
#                 break
#         prob = float(prob) #convert probability from string to float
#
#         #fill out each class depending on information in line read
#         if (startlocation == "Fairway"):
#             Fairway.setProb(prob, action, endlocation)
#         elif (startlocation == "Ravine"):
#             Ravine.setProb(prob, action, endlocation)
#         elif (startlocation == "Close"):
#             Close.setProb(prob, endlocation)
#         elif (startlocation == "Same"):
#             Same.setProb(prob, endlocation)
#         elif (startlocation == "Left"):
#             Left.setProb(prob, endlocation)
#         elif (startlocation == "In"):
#             In.setProb(prob, endlocation)
#         elif (startlocation == "Over"):
#             Over.setProb(prob, action, endlocation)
#     #if entry is '\n', data has ended
#     if entry == '\n':
#         break

#print(f'Fairway: {Fairway.printState()}')
#print(f'Ravine: {Ravine.printState()}')
#print(f'Close: {Close.printState()}')
#print(f'State: {Same.printState()}')
#print(f'Left: {Left.printState()}')
#print(f'In: {In.printState()}')
#print(f'Over: {Over.printState()}')

#dictionareis to hash indexes to states/actions
states = {}
actions = {}


#define states
fairway = Fairway1()
ravine = Ravine1()
leftOfPin = LeftOfThePin()
closeToPin = CloseToPin()
inTheHole = InTheHole()
sameLevel = SameLevelAsPin()
overTheGreen = OverTheGreen()

#define actions and add to dictionaries
atPin = AtPin(1)
states[atPin.getNumber()] = atPin
pastPin = PastPin(2)
states[pastPin.getNumber()] = pastPin
leftPin = LeftPin(3)
states[leftPin.getNumber()] = leftPin
chip = Chip(4)
states[chip.getNumber()] = chip
pitch = Pitch(5)
states[pitch.getNumber()] = pitch
putt = Putt(6)
states[putt.getNumber()] = putt


#hash state numbers to probabilites
dict1 = {}
dict1[closeToPin.getNumber()] = .25
dict1[sameLevel.getNumber()] = .35
dict1[ravine.getNumber()] = .15
dict1[leftOfPin.getNumber()] = .1
dict1[overTheGreen.getNumber()] = .15

#create connectiong beween the probabilties of moving to a new states from a previous state
atPin.setStates(fairway.getNumber(), dict1)

dict1 = {}
dict1[closeToPin.getNumber()] = .2
dict1[sameLevel.getNumber()] = .35
dict1[ravine.getNumber()] = .15
dict1[leftOfPin.getNumber()] = .2
dict1[overTheGreen.getNumber()] = .1

atPin.setStates(ravine.getNumber(), dict1)

dict1 = {}
dict1[ravine.getNumber()] = .02
dict1[closeToPin.getNumber()] = .18
dict1[sameLevel.getNumber()] = .5
dict1[leftOfPin.getNumber()] = .1
dict1[overTheGreen.getNumber()] = .2

pastPin.setStates(fairway.getNumber(),  dict1)

dict1 = {}
dict1[ravine.getNumber()] = .02
dict1[closeToPin.getNumber()] = .05
dict1[sameLevel.getNumber()] = .6
dict1[leftOfPin.getNumber()] = .18
dict1[overTheGreen.getNumber()] = .15

pastPin.setStates(ravine.getNumber(), dict1)

dict1 = {}
dict1[ravine.getNumber()] = .1
dict1[closeToPin.getNumber()] = .05
dict1[sameLevel.getNumber()] = .2
dict1[leftOfPin.getNumber()] = .5
dict1[overTheGreen.getNumber()] = .15

leftPin.setStates(fairway.getNumber(), dict1)

dict1 = {}
dict1[ravine.getNumber()] = .08
dict1[closeToPin.getNumber()] = .02
dict1[sameLevel.getNumber()] = .2
dict1[leftOfPin.getNumber()] = .6
dict1[overTheGreen.getNumber()] = .1

leftPin.setStates(ravine.getNumber(), dict1)

dict1 = {}
dict1[ravine.getNumber()] = .09
dict1[closeToPin.getNumber()] = .3
dict1[sameLevel.getNumber()] = .3
dict1[leftOfPin.getNumber()] = .2
dict1[overTheGreen.getNumber()] = .1
dict1[inTheHole.getNumber()] = .01

chip.setStates(overTheGreen.getNumber(), dict1)

dict1 = {}
dict1[ravine.getNumber()] = .04
dict1[closeToPin.getNumber()] = .4
dict1[sameLevel.getNumber()] = .4
dict1[leftOfPin.getNumber()] = .1
dict1[overTheGreen.getNumber()] = .04
dict1[inTheHole.getNumber()] = .02

pitch.setStates(overTheGreen.getNumber(), dict1)

dict1 = {}
dict1[closeToPin.getNumber()] = .75
dict1[sameLevel.getNumber()] = .2
dict1[leftOfPin.getNumber()] = .04
dict1[inTheHole.getNumber()] = .01

putt.setStates(sameLevel.getNumber(), dict1)

dict1 = {}
dict1[closeToPin.getNumber()] = .49
dict1[sameLevel.getNumber()] = .3
dict1[leftOfPin.getNumber()] = .2
dict1[inTheHole.getNumber()] = .01

putt.setStates(leftOfPin.getNumber(), dict1)

dict1 = {}
dict1[closeToPin.getNumber()] = .15
dict1[inTheHole.getNumber()] = .85

putt.setStates(closeToPin.getNumber(), dict1)

#create pssible lists of actions for state
fairway.addToActions(pastPin)
fairway.addToActions(leftPin)
fairway.addToActions(atPin)

ravine.addToActions(pastPin)
ravine.addToActions(leftPin)
ravine.addToActions(atPin)

closeToPin.addToActions(putt)
leftOfPin.addToActions(putt)
sameLevel.addToActions(putt)

overTheGreen.addToActions(chip)
overTheGreen.addToActions(pitch)

#set goal
goal = inTheHole

#used to create M and Q. association between states and action that can be taken at each state
edges = [(fairway, atPin), (fairway, leftPin), (fairway, pastPin), (ravine, atPin),
         (ravine, leftPin), (ravine, pastPin), (leftOfPin, putt), (closeToPin, putt), (sameLevel, putt),
         (overTheGreen, chip), (overTheGreen, pitch)]

MATRIX_SIZE = 7
M = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
M *= -1

#populate M matrix with reward for each state
for point in edges:
    print(point)
    if point[1] is goal:
        M[point[0].getNumber(), point[1].getNumber()] = 100
    elif (point[0] is fairway or point[0] is ravine) and (point[1] is atPin or point[1] is leftPin or point[1] is pastPin):
        M[point[0].getNumber(), point[1].getNumber()] = .5
    elif (point[0] is fairway or point[0] is ravine) and point[1] is closeToPin:
        M[point[0].getNumber(), point[1].getNumber()] = 1
    elif point[0] is closeToPin and (point[1] is leftOfPin or point[1] is overTheGreen or point[1] is sameLevel):
        M[point[0].getNumber(), point[1].getNumber()] = -.5
    elif point[0] is overTheGreen and point[1] is leftOfPin or point[1] is sameLevel:
        M[point[0].getNumber(), point[1].getNumber()] = .5
    elif point[0] is sameLevel and point[1] is overTheGreen or point[1] is leftOfPin:
        M[point[0].getNumber(), point[1].getNumber()] = .5
    elif point[0] is leftOfPin and point[1] is overTheGreen:
        M[point[0].getNumber(), point[1].getNumber()] = .5
    else:
        M[point[0].getNumber(), point[1].getNumber()] = 1

#set goal reward
M[goal.getNumber(), goal.getNumber()] = 100
print('\n')
print(M)

Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

#discount paramter
gamma = 0.9

#learning rate
alpha = 1

# start on the fairway
initial_state = 1

# Determines the available actions for a given state
def available_actions(state):
    if state == 0 or state == 1:
        available_action = fairway.getActions()
    elif state == 3:
        available_action = overTheGreen.getActions()
    elif state == 6:
        return None
    else:
        available_action = sameLevel.getActions()
    #current_state_row = M[state,]
    #available_action = np.where(current_state_row >= 0)[1]

    #returns a list of action objects
    return available_action

available_action = available_actions(initial_state)
# nextState = takeAction(initial_state, available_action)
# utility = utility(initial_state, )

#Chooses one of the available actions. Epsilon controls the ratio between exploration and exploitation. Higher epsilon,
#more exploration
def sample_next_action(available_actions_range):
    epsilon = .5
    if random.uniform(0, 1) < epsilon:
        next_action = np.random.choice(available_action, 1)
    else:
        tmp = np.max(Q, axis=1).shape[0] - 1
        next_action = states[tmp]
        next_action = [next_action]
    #next_action[0].getProb()
    return next_action

action = sample_next_action(available_action)  #say next action is chip with value 1, action = 1


# Updates the Q-Matrix according to the path chosen
def update(current_state, action, gamma):
    try:
        max_index = np.where(Q[action[0].getNumber(),] == np.max(Q[action[0].getNumber(),]))[1]
        if max_index.shape[0] > 1:
            max_index = int(np.random.choice(max_index, size=1))
        else:
            max_index = int(max_index)
        max_value = Q[action[0].getNumber(), max_index]*alpha
                    # *action[0].getProb(current_state)[current_state]
        Q[current_state, action[0].getNumber()] = round(M[current_state, action[0].getNumber()] + gamma * max_value, 3)
        if (np.max(Q) > 0):
            return (np.sum(Q / np.max(Q)))
        else:
            return (0)
    except KeyError:
        print(current_state)
        print(action)

#ignore
def updateAll(current_state, action, gamma):
    try:
        row = 0
        column = 0
        for x in np.nditer(M):
            while (row < Q.shape[1]):
                while (column < Q.shape[0]):
                    value = Q[row, column]
                    Q[row, column] = x + gamma * value
                    column = column + 1
                row = row + 1
                column = 0
        if (np.max(Q) > 0):
            return (np.sum(Q / np.max(Q)))
        else:
            return (0)
    except KeyError:
        print(current_state)
        print(action)
    # Updates the Q-Matrix according to the path chosen

update(initial_state, action, gamma)

scores = []
epsilon = 1000

#run 1000 iterations of the golf game
for i in range(epsilon):
    current_state = np.random.randint(2, int(Q.shape[0]))
    available_action = available_actions(current_state)
    if available_action is None:
        #scores.append(3.5)
        continue
    action = sample_next_action(available_action)
    score = update(current_state, action, gamma)
    scores.append(score)

#show trained matrix. Matrix contains the reqard/utilites for each state/action combination
print("Trained Q matrix:")
print(Q / np.max(Q))

# Testing
current_state = 0
steps = [current_state]

#generate the best path to take
while current_state != 6:
    print(current_state)
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)
    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:")
print(steps)

#show graph of the score acheived over time
pl.plot(scores)
pl.xlabel('No of iterations')
pl.ylabel('Reward gained')
pl.show()

#attmept to go through entire Q matrix, but doesnt work for mysterious reasons
# for i, row in enumerate(Q):
#     for j, column in enumerate(Q):
#         print(f'Combination for state {(j)}, action{(i)}: {Q.item((j, i))}')
