#http://firsttimeprogrammer.blogspot.co.uk/2016/09/getting-ai-smarter-with-q-learning.html

import numpy as np

# Tasty rewards matrix
R = np.matrix([[-1, -1, -1, -1, 0, -1],
               [-1, -1, -1, 0, -1, 100],
               [-1, -1, -1, 0, -1, -1],
               [-1, 0, 0, -1, 0, -1],
               [-1, 0, 0, -1, -1, 100],
               [-1, 0, -1, -1, 0, 100]])

# Q (Quality) matrix
Q = np.matrix(np.zeros([6, 6]))

print(Q)
print(R)

# Gamma, learning paramaeter
gamma = 0.8

# Initial state (usually chosen to be random)
initState = 1

# Get all the avaliable actions avaliable to the current state
def getAvaliableActions(state):
    currentStateRow = R[state, ]
    avActions = np.where(currentStateRow >= 0)[1]
    return avActions

# Get the avalaible actions for the current state
avActions = getAvaliableActions(initState)

# Choose a random action that can be performed (so only avaliable actions)
def chooseAction(avActions):
    nextAction = int(np.random.choice(avActions, 1))
    return nextAction

# Sample the next action to be performed
action = chooseAction(avActions)

# Update the Q matrix according to the equation for the Q learning algorithm
def update(currState, action, gamma):
    # Get the indexs of which choices have the highest quality
    maxIndex = np.where(Q[action, ] == np.max(Q[action, ]))[1]

    if maxIndex.shape[0] > 1:
        # More than once choice with the highest quality
        # Choose one of them randomly
        maxIndex = int(np.random.choice(maxIndex, size = 1))
    else:
        maxIndex = int(maxIndex)

    maxValue = Q[action, maxIndex]

    # Q-Learning formula
    # Quality value of this state is the current reward, plus a weighted long term reward
    Q[currState, action] = R[currState, action] + gamma * maxValue


update(initState, action, gamma)

# Train the model, 10,000 iterations
# I think this starts in just random rooms and tries to find just the next step to take
for i in range(10000):
    currState = np.random.randint(0, int(Q.shape[0]))
    avActions = getAvaliableActions(currState)
    action = chooseAction(avActions)
    update(currState, action, gamma)

# Normalise the quality matrix
print("Trained Q matrix: ");
print(Q/np.max(Q) * 100)

# Test the trained matrix

# Goal state = 5
# Best sequence path starting from 2 -> 2, 3, 1, 5

currState = 2
steps = [currState]

while currState != 5:
    # Get the next step with the best qaulity value
    nextStepChoices = np.where(Q[currState, ] == np.max(Q[currState, ]))[1]

    #If there is more than one best choice avaliable
    if nextStepChoices.shape[0] > 1:
        # Then just choose a random one
        nextStepIndex = int(np.random.choice(nextStepChoices, size = 1))
    else:
        nextStepIndex = int(nextStepChoices)

    steps.append(nextStepIndex)
    currState = nextStepIndex

# Print the selection of steps
print("Selected path: ")
print(steps)

