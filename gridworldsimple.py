#http://outlace.com/rlpart3.html

import numpy as np

# Return two random ints, between s (inclusive) and e (exsclusive)
def randPair(s, e):
    return np.random.randint(s, e), np.random.randint(s, e)

# FInds an array in the "depth" dimension of the grid
def findLoc(state, obj):
    for i in range (0, 4):
        for j in range(0, 4):
            if (state[i, j] == obj).all():
                return i,j

# Initialise the grid deterministically
# 2d grid with one-hot encoded object in the position 
def initGrid():
    state = np.zeros((4, 4, 4))
    # Place the player
    state[0, 1] = np.array([0,0,0,1])
    # Place the wall
    state[2, 2] = np.array([0,0,1,0])
    # Place the pit
    state[1, 1] = np.array([0,1,0,0])
    # Place the player
    state[3, 3] = np.array([1,0,0,0])
    return state

# Initialise player in a random location, but keep everything else in the same place
def initGridPlayer():
    state = np.zeros((4, 4, 4))
    # Place the player
    state[randPair(0, 4)] = np.array([0,0,0,1])
    # Place the wall
    state[2, 2] = np.array([0,0,1,0])
    # Place the pit
    state[1, 1] = np.array([0,1,0,0])
    # Place the player
    state[3, 3] = np.array([1,0,0,0])

    #Find the grid position of the player (agent)
    a = findLoc(state, np.array([0,0,0,1]))
    # fine wall
    w = findLoc(state, np.array([0,0,1,0]))
    # find goal
    g = findLoc(state, np.array([1,0,0,0]))
    # find the pit
    p = findLoc(state, np.array([0,1,0,0]))

    if (not a or not w or not g or not p):
        #Invalid grid, rebuilding...
        return initGridPlayer()

    return state

# Initiase grid so that goal, pit, wall, player are all randomly placed
def initGridRand():
    state = np.zeros((4, 4, 4))
    # Place the player
    state[randPair(0, 4)] = np.array([0,0,0,1])
    # Place wall
    state[randPair(0, 4)] = np.array([0,0,1,0])
    # Place pit
    state[randPair(0, 4)] = np.array([0,1,0,0])
    # Place goal
    state[randPair(0, 4)] = np.array([1,0,0,0])
    
    #Find the grid position of the player (agent)
    a = findLoc(state, np.array([0,0,0,1]))
    # fine wall
    w = findLoc(state, np.array([0,0,1,0]))
    # find goal
    g = findLoc(state, np.array([1,0,0,0]))
    # find the pit
    p = findLoc(state, np.array([0,1,0,0]))

    if (not a or not w or not g or not p):
        #Invalid grid, rebuilding...
        return initGridPlayer()

    return state

# Implement the movement function
def makeMove(state, action):
    # Need to locate the player in the grid
    # Need to determine what object (if any) is in the new grid spot the player wants to move to 
    playerLoc = findLoc(state, np.array([0,0,0,1]))
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))

    # Reset the map, we're going to place everything back again
    state = np.zeros((4, 4, 4))

    # You can move one of the four cardinal directions
    actions = [[-1,0],[1,0],[0,-1],[0,1]]

    newLoc = (playerLoc[0] + actions[action][0], playerLoc[1] + actions[action][1])

    if newLoc != wall:
        # I think this next line is checking that we don't go over the edge of the grid
        if (newLoc[0] >= 0 and newLoc[1] >= 0 and newLoc[0] <= 3 and newLoc[1] <= 3):
            # Set this new location to be where the player is
            state[newLoc][3] = 1

    # Find out where the player moved to 
    newPlayerLoc = findLoc(state, np.array([0,0,0,1]))
    if (not newPlayerLoc):
        state[playerLoc] = np.array([0,0,0,1])
    # Re place pit
    state[pit][1] = 1
    # Re plae wall
    state[wall][2] = 1
    # Re place goal
    state[goal][0] = 1

    return state

# Gets the location of an object, but can see it even if two
# Objects are on the same space
def getLoc(state, level):
    for i in range(0, 4):
        for j in range(0, 4):
            if (state[i, j][level] == 1):
                return i,j

# Calculate the reward for moving to this square
def getReward(state):
    playerLoc = getLoc(state, 3)
    pit = getLoc(state, 1)
    goal = getLoc(state, 0)

    if playerLoc == pit:
        return -10
    elif playerLoc == goal:
        return 10
    else:
        return -1

# Display the gridworld grid to the cmd
def dispGrid(state):
    grid = np.zeros((4,4), dtype=str)
    playerLoc = findLoc(state, np.array([0,0,0,1]))
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))

    for i in range(0, 4):
        for j in range(0,4):
            grid[i, j] = ' '

    # Add all the objects to the grid to display
    if playerLoc:
        grid[playerLoc] = 'p'
    if wall:
        grid[wall] = 'W'
    if goal:
        grid[goal] = '+'
    if pit:
        grid[pit] = '-'

    return grid

state = initGridRand()
print(dispGrid(state))

# Start the neueal network Q learning, using keras with tensorflow backend

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

# Just disables the warning, doesn't enable AVX/FMA
# TODO: Maybe try and install a build so they can be used
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = Sequential()
model.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(64, )))
model.add(Activation('relu'))

model.add(Dense(150, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))

model.add(Dense(4, kernel_initializer='lecun_uniform'))
# Use linear output so we can have a range of real-valued outputs
model.add(Activation('linear'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

# Just to shown an example output, read outputs left to right, up/down/left/right
print(model.predict(state.reshape(1, 64), batch_size=1))

# Now let us train the model
from IPython.display import clear_output
import random

#epochs = 1000
epochs = 0000
# Since it may take several moves to the goal, making gamma high
gamma = 0.9 
epsilon = 1

# Run for 1000 games
for i in range(epochs):

    # Initialise tha absolutely static grid
    state = initGrid()
    # Is the game still running?
    status = 1

    # Run while the game is still in progress
    while status == 1:
        # We are in state S
        # Let us run our Q function on the S to get the qulality values for all the possible actions
        # This returns 4 values for left/right/down/up
        qval = model.predict(state.reshape(1, 64), batch_size=1)

        # Perform epislon greedy to choose how to do the next function
        # epislon is decreased for every game

        if (random.random() < epsilon):
            # Choose a random action
            action = np.random.randint(0, 4)
        else:
            # Choose the best action from the Q(s, a) function
            action = (np.argmax(qval))

        # Take this chosen action, observe the next state S'
        newState = makeMove(state, action)
        # Observe the reward
        reward = getReward(newState)
        # Get the max Q(S', a)
        newQ = model.predict(newState.reshape(1, 64), batch_size=1)
        maxQ = np.max(newQ)

        # Create matrix y same dimentions 1,4 and copy the quality values into it
        y = np.zeros((1,4))
        y[:] = qval[:]

        if reward == -1:
            # Non-terminal state
            update = (reward + (gamma * maxQ))
        else:
            # Terminal state
            update = reward

        # Target output, teach the quality value
        # Set the action value to be the reward, which is calcualted from the main equation
        y[0][action] = update

        print("Game #: %s" % (i, ))

        # Train the CNN with the results array that was created in y
        model.fit(state.reshape(1, 64), y, batch_size=1, nb_epoch=1, verbose=1)
        state = newState

        # Carry on as long as we havent got to the pit or the goal
        if reward != -1:
            status = 0

        # I think this does not work as I am in a terminal and not a python terminal
        clear_output(wait=True)

    # Decrease the liklehood of taking a random movement over time
    if epsilon > 0.1:
        epsilon -= (1/epochs)

# Test the neural network to see if it actually learned how to solve the solution
def testAlgo(init=0):
    i = 0

    if init == 0:
        state = initGrid()
    elif init == 1:
        state = initGridPlayer()
    elif init == 2:
        state = initGridRand()

    print("Initial State:")
    print(dispGrid(state))
    status = 1

    # Keep going while the game is in progress
    while status == 1:
        qval = model.predict(state.reshape(1,64), batch_size=1)
        # Take the action with the highest quality value
        action = (np.argmax(qval))
        print('Move #: %s; Taking action %s' % (i, action))
        state = makeMove(state, action)
        print(dispGrid(state))

        reward = getReward(state)

        if reward != -1:
            status = 0
            print("Reward: %s" % (reward, ))

        i += 1

        # If we're taking more than 10 actions, then just stop
        if i > 10:
            print("Game lost; too many moves.")
            break

#testAlgo(init=0)

# Reset the weights of the NN
model.compile(loss='mse', optimizer=rms)
epochs = 3000
gamma = 0.975
epsilon = 1
batchSize = 40
buffer = 80
replay = []
# Stores tuples of (S, A, R, S')
h = 0

for i in range(epochs):
    # Using the harder state init function
    state = initGridPlayer()
    status = 1
    # While the game is still in progress
    while status == 1:
        # We are in state S
        # Let us run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1,64), batch_size=1)

        if (random.random() < epsilon):
            # Here we are choosing a random value
            action = np.random.randint(0, 4)
        else:
            # Choose best action from the quality function that takes in state and action
            action = (np.argmax(qval))

        # Take action, observe new state S'
        newState = makeMove(state, action)
        # Observe the reward from taking this movement
        reward = getReward(newState)


        # Implement the experience replay storage
        if (len(replay) < buffer):
            # The buffer is not filled, so append to it
            replay.append((state, action, reward, newState))
        else:
            # The bugger is now full, overwrite old values
            if (h < (buffer - 1)):
                h += 1
            else:
                h = 0

            # Overite index h with the data, instead of appending like before
            replay[h] = (state, action, reward, newState)

            # Randomly sample out experience replay memory
            miniBatch = random.sample(replay, batchSize)

            # State of each memory
            XTrain = []
            # Update value for each train data in the batch
            YTrain = []

            for memory in miniBatch:
                # Get the maxQ('s,a)
                # Get the old data out of the sampled memory
                oldState, action, reward, newState = memory
                # Get the quality values from that state
                oldQVal = model.predict(oldState.reshape(1, 64), batch_size=1)

                # Now get quality values from the move that was made, as usual
                newQ = model.predict(newState.reshape(1, 64), batch_size=1)
                maxQ = np.max(newQ)

                # Create the y matrix, which is the Q matrix but with the updated values
                y = np.zeros((1, 4))
                y[:] = oldQVal[:]

                if reward == -1:
                    # Non-terminal state
                    update = (reward + gamma * maxQ)
                else:
                    # Terminal state
                    update = reward

                y[0][action] = update

                XTrain.append(oldState.reshape(64,))
                YTrain.append(y.reshape(4,))

            # Turn these arrays into numpy matrices
            XTrain = np.array(XTrain)
            YTrain = np.array(YTrain)

            print("Game #: %s" % (i, ))

            # Train the model using the sampled minibatch data
            model.fit(XTrain, YTrain, batch_size=batchSize, nb_epoch=1, verbose=1)
            state = newState
        # End the experience replay storage



        if reward != -1:
            # Reached the terminal state, update game status
            status = 0
        
        clear_output(wait=True)

        if epsilon > 0.1:
            epsilon -= (1/epochs)

testAlgo(1)
