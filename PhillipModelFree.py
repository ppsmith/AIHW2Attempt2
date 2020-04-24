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
import pylab as pl

# dictionareis to hash indexes to states/actions
states = {}
actions = {}

# define states
fairway = Fairway1()
states["fairway".upper()] = fairway
ravine = Ravine1()
states["ravine".upper()] = ravine
leftOfPin = LeftOfThePin()
states["left".upper()] = leftOfPin
closeToPin = CloseToPin()
states["close".upper()] = closeToPin
inTheHole = InTheHole()
states["in".upper()] = inTheHole
sameLevel = SameLevelAsPin()
states["same".upper()] = sameLevel
overTheGreen = OverTheGreen()
states["over".upper()] = overTheGreen

# define actions and add to dictionaries
atPin = AtPin(0)
states[atPin.getNumber()] = atPin
pastPin = PastPin(1)
states[pastPin.getNumber()] = pastPin
leftPin = LeftPin(2)
states[leftPin.getNumber()] = leftPin
chip = Chip(3)
states[chip.getNumber()] = chip
pitch = Pitch(4)
states[pitch.getNumber()] = pitch
putt = Putt(5)
states[putt.getNumber()] = putt

dictFairwayAt = {}
dictFairwayPast = {}
dictFairwayLeft = {}
dictRavineAt = {}
dictRavinePast = {}
dictRavineLeft = {}
dictClosePutt = {}
dictSamePutt = {}
dictLeft = {}
dictOverChip = {}
dictOverPitch = {}

data = open(input('Please give file path'), 'r')
#
# #read each line of data
for entry in data:
    # make sure that data hasn't ended
    if entry != '\n':
        # Each line goes start location, action, end location, and probability of ending in end state
        # Each aspect is separated by a /, so find positions of /
        sub = entry
        startlocation = "invalid"
        action = "invalid"
        endlocation = "invalid"
        prob = "0"
        current_index = 0
        num_slash = 0
        #         #break up each line to find the start location, the action, the end location, and the probability
        while (current_index != -1):
            current_index = sub.find("/")
            # print(current_index)
            num_slash = num_slash + 1
            if (num_slash == 1):  # start location
                startlocation = sub[0:current_index]
                sub = sub[current_index + 1:len(sub)]  # create substring starting from action
            elif (num_slash == 2):  # action
                action = sub[0:current_index]  # create substring starting from end location
                sub = sub[current_index + 1:len(sub)]
            elif (num_slash == 3):  # end location and probability
                endlocation = sub[0:current_index]
                prob = sub[current_index + 2:len(sub)]
                sub = sub[current_index + 1:len(sub)]
            else:
                break
        prob = float(prob)  # convert probability from string to float
        if len(str(prob)) <= 3:
            prob = prob / 10
        else:
            prob = prob / 100
        # fill out each class depending on information in line read
        if (startlocation == "Fairway") and action == "At":
            dictFairwayAt[states[endlocation.upper()].getNumber()] = prob
            fairway.addToActions(atPin)
        elif (startlocation == "Fairway") and action == "Past":
            dictFairwayPast[states[endlocation.upper()].getNumber()] = prob
            fairway.addToActions(pastPin)
        elif (startlocation == "Fairway") and action == "Left":
            dictFairwayLeft[states[endlocation.upper()].getNumber()] = prob
            fairway.addToActions(leftPin)
        elif (startlocation == "Ravine") and action == "At":
            dictRavineAt[states[endlocation.upper()].getNumber()] = prob
            ravine.addToActions(atPin)
        elif (startlocation == "Ravine") and action == "Past":
            dictRavinePast[states[endlocation.upper()].getNumber()] = prob
            ravine.addToActions(pastPin)
        elif (startlocation == "Ravine") and action == "Left":
            dictRavineLeft[states[endlocation.upper()].getNumber()] = prob
            ravine.addToActions(leftPin)
        elif (startlocation == "Close") and action == "Putt":
            dictClosePutt[states[endlocation.upper()].getNumber()] = prob
            closeToPin.addToActions(putt)
        elif (startlocation == "Same") and action == "Putt":
            dictSamePutt[states[endlocation.upper()].getNumber()] = prob
            sameLevel.addToActions(putt)
        elif (startlocation == "Left"):
            dictLeft[states[endlocation.upper()].getNumber()] = prob
            leftOfPin.addToActions(putt)
        elif (startlocation == "Over") and action == "Chip":
            dictOverChip[states[endlocation.upper()].getNumber()] = prob
            overTheGreen.addToActions(chip)
        elif (startlocation == "Over") and action == "Pitch":
            dictOverPitch[states[endlocation.upper()].getNumber()] = prob
            overTheGreen.addToActions(pitch)
    # if entry is '\n', data has ended
    if entry == '\n':
        break

# create connectiong beween the probabilties of moving to a new states from a previous state
atPin.setStates(fairway.getNumber(), dictFairwayAt)

atPin.setStates(ravine.getNumber(), dictRavineAt)

pastPin.setStates(fairway.getNumber(), dictFairwayPast)

pastPin.setStates(ravine.getNumber(), dictRavinePast)

leftPin.setStates(fairway.getNumber(), dictFairwayLeft)

leftPin.setStates(ravine.getNumber(), dictRavineLeft)

chip.setStates(overTheGreen.getNumber(), dictOverChip)

pitch.setStates(overTheGreen.getNumber(), dictOverPitch)

putt.setStates(sameLevel.getNumber(), dictSamePutt)

putt.setStates(leftOfPin.getNumber(), dictLeft)

putt.setStates(closeToPin.getNumber(), dictClosePutt)

putt.setStates(inTheHole.getNumber(), dictClosePutt)

# set goal
goal = inTheHole

# used to create M and Q. association between states and action that can be taken at each state
edges = [(fairway, atPin), (fairway, leftPin), (fairway, pastPin), (ravine, atPin),
         (ravine, leftPin), (ravine, pastPin), (leftOfPin, putt), (closeToPin, putt), (sameLevel, putt),
         (overTheGreen, chip), (overTheGreen, pitch)]

MATRIX_SIZE = 7
M = np.matrix(np.ones(shape=(7, 6)))
M *= -1

# populate M matrix with reward for each state
for point in edges:
    print(point)
    if point[1] is goal:
        M[point[0].getNumber(), point[1].getNumber()] = 100
    elif (point[0] is fairway or point[0] is ravine) and (
            point[1] is atPin or point[1] is leftPin or point[1] is pastPin):
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

# set goal reward
M[goal.getNumber(), goal.getNumber() - 1] = 100
print('\n')
print(M)

Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

# discount paramter
gamma = 0.9

# learning rate
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
    # current_state_row = M[state,]
    # available_action = np.where(current_state_row >= 0)[1]

    # returns a list of action objects
    return available_action


available_action = available_actions(initial_state)


# nextState = takeAction(initial_state, available_action)
# utility = utility(initial_state, )

# Chooses one of the available actions. Epsilon controls the ratio between exploration and exploitation. Higher epsilon,
# more exploration
def sample_next_action(available_actions_range, current_state):
    epsilon = .9
    if random.uniform(0, 1) < epsilon:
        next_action = np.random.choice(available_action, 1)
    else:
        try:
            next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
            # tmp = np.max(Q, axis=1).shape[0] - 1
            next_action = states[next_step_index[0]]
            next_action = [next_action]
        except KeyError:
            return None
    # next_action[0].getProb()
    return next_action


action = sample_next_action(available_action, initial_state)  # say next action is chip with value 1, action = 1


# Updates the Q-Matrix according to the path chosen
def update(current_state, action, gamma):
    try:
        max_index = np.where(Q[action[0].getNumber(),] == np.max(Q[action[0].getNumber(),]))[1]
        if max_index.shape[0] > 1:
            max_index = int(np.random.choice(max_index, size=1))
        else:
            max_index = int(max_index)
        max_value = Q[action[0].getNumber(), max_index] * alpha * action[0].getProb(current_state)[current_state]
        Q[current_state, action[0].getNumber()] = round(M[current_state, action[0].getNumber()] + gamma * max_value, 3)
        if (np.max(Q) > 0):
            return (np.sum(Q / np.max(Q)))
        else:
            return (0)
    except KeyError:
        print(current_state)
        print(action)


# ignore
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
epsilon = 500

# run epsilon iterations of the golf game
for i in range(epsilon):
    current_state = np.random.randint(2, int(Q.shape[0]))
    if (current_state == 6):
        continue
    available_action = available_actions(current_state)
    # if available_action is None:
    #     #scores.append(3.5)
    #     continue
    action = sample_next_action(available_action, current_state)
    score = update(current_state, action, gamma)
    scores.append(score)

# show trained matrix. Matrix contains the reqard/utilites for each state/action combination
print("Trained Q matrix:")
print(Q / np.max(Q))

# Testing
current_state = 0
steps = [current_state]

# generate the best path to take
while current_state != 5:
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

# show graph of the score acheived over time
pl.plot(scores)
pl.xlabel('No of iterations')
pl.ylabel('Reward gained')
pl.show()
