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
#import pylab as pl

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
    #print(point)
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
#print('\n')
#print(M)

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

trial_matrix = np.matrix(np.ones(shape=(epsilon, 2)))

# run epsilon iterations of the golf game
for i in range(epsilon):
    current_state = np.random.randint(2, int(Q.shape[0]))
    trial_matrix[i, 0] = current_state
    if (current_state == 6):
        continue
    available_action = available_actions(current_state)
    # if available_action is None:
    #     #scores.append(3.5)
    #     continue
    action = sample_next_action(available_action, current_state)
    trial_matrix[i, 1] = action[0].getNumber()
    score = update(current_state, action, gamma)
    scores.append(score)

#states:
#fairway = 0
#ravine = 1
#leftOfPin = 2
#closeToPin = 5
#inHole = 6
#sameLevel = 4
#overTheGreen = 3

#actions:
#atPin = 1
#pastPin = 2
#leftPin = 3
#chip = 4
#pitch = 5
#putt = 6

#statecounts
numfairway = 0
numravine = 0
numleftofpin = 0
numclosetopin = 0
numinhole = 0
numsamelevel = 0
numoverthegreen = 0

count01 = 0 #fairway-at
count015 = 0 #fairway-at-close
count014 = 0 #fairway-at-same
count013 = 0 #fairway-at-over
count011 = 0 #fairway-at-ravine
count012 = 0 #fairway-at-left

count02 = 0 #fairway-past
count025 = 0 #fairway-past-close
count024 = 0 #fairway-past-same
count023 = 0 #fairway-past-over
count021 = 0 #fairway-past-ravine
count022 = 0 #fairway-past-left

count03 = 0 #fairway-left
count035 = 0 #fairway-left-close
count034 = 0 #fairway-left-same
count033 = 0 #fairway-left-over
count031 = 0 #fairway-left-ravine
count032 = 0 #fairway-left-left

count11 = 0 #ravine-at
count115 = 0 #ravine-at-close
count111 = 0 #ravine-at-ravine
count112 = 0 #ravine-at-left
count114 = 0 #ravine-at-same
count113 = 0 #ravine-at-over

count12 = 0 #ravine-past
count125 = 0 #ravine-past-close
count121 = 0 #ravine-past-ravine
count122 = 0 #ravine-past-left
count124 = 0 #ravine-past-same
count123 = 0 #ravine-past-over

count13 = 0 #ravine-left
count135 = 0 #ravine-at-close
count131 = 0 #ravine-at-ravine
count132 = 0 #ravine-at-left
count134 = 0 #ravine-at-same
count133 = 0 #ravine-at-over

count34 = 0 #over-chip
count345 = 0 #over-chip-close
count344 = 0 #over-chip-same
count341 = 0 #over-chip-ravine
count343 = 0 #over-chip-over
count342 = 0 #over-chip-left
count346 = 0 #over-chip-in

count35 = 0 #over-pitch
count355 = 0 #over-pitch-close
count354 = 0 #over-pitch-same
count351 = 0 #over-pitch-ravine
count353 = 0 #over-pitch-over
count352 = 0 #over-pitch-left
count356 = 0 #over-pitch-in

count46 = 0 #same-putt
count464 = 0 #same-putt-same
count465 = 0 #same-putt-close
count462 = 0 #same-putt-left
count466 = 0 #same-putt-in

count26 = 0 #left-putt
count264 = 0 #left-putt-same
count265 = 0 #left-putt-close
count262 = 0 #left-putt-left
count266 = 0 #left-putt-in

count56 = 0 #close-putt
count565 = 0 #close-putt-close
count566 = 0 #close-putt-in

initial_states_for_table = np.asmatrix([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[2],[2],[2],[2],[3],[3],[3],[3],[3],[3],[3],[3],[3],[3],[3],[3],[4],[4],[4],[4],[5],[5]])
actions_for_table= np.asmatrix([[1],[1],[1],[1],[1],[2],[2],[2],[2],[2],[3],[3],[3],[3],[3],[1],[1],[1],[1],[1],[2],[2],[2],[2],[2],[3],[3],[3],[3],[3],[6],[6],[6],[6],[4],[4],[4],[4],[4],[4],[5],[5],[5],[5],[5],[5],[6],[6],[6],[6],[6],[6]])
end_states_for_table= np.asmatrix([[1],[2],[3],[4],[5],[1],[2],[3],[4],[5],[1],[2],[3],[4],[5],[1],[2],[3],[4],[5],[1],[2],[3],[4],[5],[1],[2],[3],[4],[5],[2],[4],[5],[6],[1],[2],[3],[4],[5],[6],[1],[2],[3],[4],[5],[6],[2],[4],[5],[6],[5],[6]])

trans_prob_table = np.matrix(np.ones(shape=(len(end_states_for_table[:,0]), 4)))

#count each time state occurs

for i in range(epsilon-1):
    if trial_matrix[i, 0] == 0: #start state
        numfairway += 1
        if trial_matrix[i, 1] == 1: #action
            count01 += 1
            if trial_matrix[i+1, 0] == 1: #end state
                count011 += 1
            elif trial_matrix[i+1, 0] == 2:
                count012 += 1
            elif trial_matrix[i+1, 0] == 3:
                count013 += 1
            elif trial_matrix[i+1, 0] == 4:
                count014 += 1
            elif trial_matrix[i+1, 0] == 5:
                count015 += 1
        elif trial_matrix[i, 1] == 2:
            count02 += 1
            if trial_matrix[i + 1, 0] == 1:  #end state
                count021 += 1
            elif trial_matrix[i + 1, 0] == 2:
                count022 += 1
            elif trial_matrix[i + 1, 0] == 3:
                count023 += 1
            elif trial_matrix[i + 1, 0] == 4:
                count024 += 1
            elif trial_matrix[i + 1, 0] == 5:
                count025 += 1
        if trial_matrix[i, 1] == 3:
            count03 += 1
            if trial_matrix[i + 1, 0] == 1:  # end state
                count031 += 1
            elif trial_matrix[i + 1, 0] == 2:
                count032 += 1
            elif trial_matrix[i + 1, 0] == 3:
                count033 += 1
            elif trial_matrix[i + 1, 0] == 4:
                count034 += 1
            elif trial_matrix[i + 1, 0] == 5:
                count035 += 1
    elif trial_matrix[i, 0] == 1:
        numravine += 1
        if trial_matrix[i, 1] == 1:
            count11 += 1
            if trial_matrix[i+1, 0] == 1:
                count111 += 1
            elif trial_matrix[i+1, 0] == 2:
                count112 += 1
            elif trial_matrix[i+1, 0] == 3:
                count113 += 1
            elif trial_matrix[i+1, 0] == 4:
                count114 += 1
            elif trial_matrix[i+1, 0] == 5:
                count115 += 1
        elif trial_matrix[i, 1] == 2:
            count12 += 1
            if trial_matrix[i+1, 0] == 1:
                count121 += 1
            elif trial_matrix[i+1, 0] == 2:
                count122 += 1
            elif trial_matrix[i+1, 0] == 3:
                count123 += 1
            elif trial_matrix[i+1, 0] == 4:
                count124 += 1
            elif trial_matrix[i+1, 0] == 5:
                count125 += 1
        elif trial_matrix[i, 1] == 3:
            count13 += 1
            if trial_matrix[i+1, 0] == 1:
                count131 += 1
            elif trial_matrix[i+1, 0] == 2:
                count132 += 1
            elif trial_matrix[i+1, 0] == 3:
                count133 += 1
            elif trial_matrix[i+1, 0] == 4:
                count134 += 1
            elif trial_matrix[i+1, 0] == 5:
                count135 += 1
    elif trial_matrix[i, 0] == 2:
        numleftofpin += 1
        if trial_matrix[i, 1] == 6:
            count26 += 1
            if trial_matrix[i+1, 0] == 2:
                count262 += 1
            elif trial_matrix[i+1, 0] == 4:
                count264 += 1
            elif trial_matrix[i+1, 0] == 5:
                count265 += 1
            elif trial_matrix[i+1, 0] == 6:
                count266 += 1
    elif trial_matrix[i, 0] == 3:
        numoverthegreen += 1
        if trial_matrix[i, 1] == 4:
            count34 += 1
            if trial_matrix[i+1, 0] == 1:
                count341 += 1
            elif trial_matrix[i+1, 0] == 2:
                count342 += 1
            elif trial_matrix[i+1, 0] == 3:
                count343 += 1
            elif trial_matrix[i+1, 0] == 4:
                count344 += 1
            elif trial_matrix[i+1, 0] == 5:
                count345 += 1
            elif trial_matrix[i+1, 0] == 6:
                count346 += 1
        elif trial_matrix[i, 1] == 5:
            count35 += 1
            if trial_matrix[i+1, 0] == 1:
                count351 += 1
            elif trial_matrix[i+1, 0] == 2:
                count352 += 1
            elif trial_matrix[i+1, 0] == 3:
                count353 += 1
            if trial_matrix[i+1, 0] == 4:
                count354 += 1
            elif trial_matrix[i+1, 0] == 5:
                count355 += 1
            elif trial_matrix[i+1, 0] == 6:
                count356 += 1
    elif trial_matrix[i, 0] == 4:
        numsamelevel += 1
        if trial_matrix[i, 1] == 6:
            count46 += 1
            if trial_matrix[i+1, 0] == 2:
                count462 += 1
            elif trial_matrix[i+1, 0] == 4:
                count464 += 1
            elif trial_matrix[i+1, 0] == 5:
                count465 += 1
            elif trial_matrix[i+1, 0] == 6:
                count466 += 1
    elif trial_matrix[i, 0] == 5:
        numclosetopin += 1
        if trial_matrix[i, 1] == 6:
            count56 += 1
            if trial_matrix[i+1, 0] == 5:
                count565 += 1
            elif trial_matrix[i+1, 0] == 6:
                count566 += 1
    elif trial_matrix[i, 0] == 6:
        numinhole += 1

trans_prob_table[:,0] = initial_states_for_table
trans_prob_table[:,1] = actions_for_table
trans_prob_table[:,2] = end_states_for_table

def findProb(B, A):
    if (A > 0):
        return B / A
    else:
        return A

for i in range(len(trans_prob_table[:,0])):
    if trans_prob_table[i,0] == 0:
        if trans_prob_table[i,1] == 1:
            if trans_prob_table[i,2] == 1:
                trans_prob_table[i,3] = findProb(count011, count01)
            elif trans_prob_table[i,2] == 2:
               trans_prob_table[i,3] = findProb(count012, count01)
            elif trans_prob_table[i,2] == 3:
                trans_prob_table[i,3] = findProb(count013, count01)
            elif trans_prob_table[i,2] == 4:
                trans_prob_table[i,3] = findProb(count014, count01)
            elif trans_prob_table[i,2] == 5:
                trans_prob_table[i,3] = findProb(count015, count01)
        elif trans_prob_table[i,1] == 2:
            if trans_prob_table[i, 2] == 1:
                trans_prob_table[i, 3] = findProb(count021, count02)
            elif trans_prob_table[i, 2] == 2:
                trans_prob_table[i, 3] = findProb(count022, count02)
            elif trans_prob_table[i, 2] == 3:
                trans_prob_table[i, 3] = findProb(count023, count02)
            elif trans_prob_table[i, 2] == 4:
                trans_prob_table[i, 3] = findProb(count024, count02)
            elif trans_prob_table[i, 2] == 5:
                trans_prob_table[i, 3] = findProb(count025, count02)
        elif trans_prob_table[i,1] == 3:
            if trans_prob_table[i,2] == 1:
                trans_prob_table[i,3] = findProb(count031, count03)
            elif trans_prob_table[i,2] == 2:
                trans_prob_table[i,3] = findProb(count032, count03)
            elif trans_prob_table[i,2] == 3:
                trans_prob_table[i,3] = findProb(count033, count03)
            elif trans_prob_table[i,2] == 4:
                trans_prob_table[i,3] = findProb(count034, count03)
            elif trans_prob_table[i,2] == 5:
                trans_prob_table[i,3] = findProb(count035, count03)
    elif trans_prob_table[i,0] == 1:
        if trans_prob_table[i,1] == 1:
            if trans_prob_table[i,2] == 1:
                trans_prob_table[i,3] = findProb(count111, count11)
            elif trans_prob_table[i,2] == 2:
                trans_prob_table[i,3] = findProb(count112, count11)
            elif trans_prob_table[i,2] == 3:
                trans_prob_table[i,3] = findProb(count113, count11)
            elif trans_prob_table[i,2] == 4:
                trans_prob_table[i,3] = findProb(count114, count11)
            elif trans_prob_table[i,2] == 5:
                trans_prob_table[i,3] = findProb(count115, count11)
        elif trans_prob_table[i,1] == 2:
            if trans_prob_table[i, 2] == 1:
                trans_prob_table[i, 3] = findProb(count121, count12)
            elif trans_prob_table[i, 2] == 2:
                trans_prob_table[i, 3] = findProb(count122, count12)
            elif trans_prob_table[i, 2] == 3:
                trans_prob_table[i, 3] = findProb(count123, count12)
            elif trans_prob_table[i, 2] == 4:
                trans_prob_table[i, 3] = findProb(count124, count12)
            elif trans_prob_table[i, 2] == 5:
                trans_prob_table[i, 3] = findProb(count125, count12)
        elif trans_prob_table[i,1] == 3:
            if trans_prob_table[i,2] == 1:
                trans_prob_table[i,3] = findProb(count131, count13)
            elif trans_prob_table[i,2] == 2:
                trans_prob_table[i,3] = findProb(count132, count13)
            elif trans_prob_table[i,2] == 3:
                trans_prob_table[i,3] = findProb(count133, count13)
            elif trans_prob_table[i,2] == 4:
                trans_prob_table[i,3] = findProb(count134, count13)
            elif trans_prob_table[i,2] == 5:
                trans_prob_table[i,3] = findProb(count135, count13)
    elif trans_prob_table[i, 0] == 2:
        if trans_prob_table[i,1] == 6:
            if trans_prob_table[i,2] == 2:
                trans_prob_table[i,3] = findProb(count262, count26)
            elif trans_prob_table[i,2] == 4:
                trans_prob_table[i,3] = findProb(count264, count26)
            elif trans_prob_table[i,2] == 5:
                trans_prob_table[i,3] = findProb(count265, count26)
            elif trans_prob_table[i,2] == 6:
                trans_prob_table[i,3] = findProb(count266, count26)
    elif trans_prob_table[i, 0] == 3:
        if trans_prob_table[i, 1] == 4:
            if trans_prob_table[i,2] == 1:
                trans_prob_table[i,3] = findProb(count341, count34)
            elif trans_prob_table[i,2] == 2:
                trans_prob_table[i,3] = findProb(count342, count34)
            elif trans_prob_table[i,2] == 3:
                trans_prob_table[i,3] = findProb(count343, count34)
            elif trans_prob_table[i,2] == 4:
                trans_prob_table[i,3] = findProb(count344, count34)
            elif trans_prob_table[i,2] == 5:
                trans_prob_table[i,3] = findProb(count345, count34)
            elif trans_prob_table[i,2] == 6:
                trans_prob_table[i,3] = findProb(count346, count34)
        elif trans_prob_table[i, 1] == 5:
            if trans_prob_table[i,2] == 1:
                trans_prob_table[i,3] = findProb(count351, count35)
            if trans_prob_table[i,2] == 2:
                trans_prob_table[i,3] = findProb(count352, count35)
            if trans_prob_table[i,2] == 3:
                trans_prob_table[i,3] = findProb(count353, count35)
            if trans_prob_table[i,2] == 4:
                trans_prob_table[i,3] = findProb(count354, count35)
            if trans_prob_table[i,2] == 5:
                trans_prob_table[i,3] = findProb(count355, count35)
            if trans_prob_table[i,2] == 6:
                trans_prob_table[i,3] = findProb(count356, count35)
    elif trans_prob_table[i, 0] == 4:
        if trans_prob_table[i, 1] == 6:
            if trans_prob_table[i, 2] == 2:
                trans_prob_table[i, 3] = findProb(count462, count46)
            elif trans_prob_table[i, 2] == 4:
                trans_prob_table[i, 3] = findProb(count464, count46)
            elif trans_prob_table[i, 2] == 5:
                trans_prob_table[i, 3] = findProb(count465, count46)
            elif trans_prob_table[i, 2] == 6:
                trans_prob_table[i, 3] = findProb(count466, count46)
    elif trans_prob_table[i, 0] == 5:
        if trans_prob_table[i, 1] == 6:
            if trans_prob_table[i, 2] == 5:
                trans_prob_table[i, 3] = findProb(count565, count56)
            if trans_prob_table[i, 2] == 6:
                trans_prob_table[i, 3] = findProb(count566, count56)


print("Transitional Probability Table")
print("[State State], [Action], [End State], [Probability Table]")
print(trans_prob_table)

# show trained matrix. Matrix contains the reqard/utilites for each state/action combination
print("Trained Q matrix:")
print(Q / np.max(Q))

# Testing
current_state = 0
steps = [current_state]

# generate the best path to take
while current_state != 5:
    #print(current_state)
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
