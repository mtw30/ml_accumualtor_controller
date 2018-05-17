# My script to process the generated data

### variables

# Q = mcDeltaT for li-ion batteries 
specHeatCap = 795.0     # Specific heat capacity (in seconds)
Tamb = float(25 + 273)  # Ambient (initial) temperature (kelin)
battMass = 80.0         # Mass of the battery in kg

# Conversion factors
mphTomps = 0.44704

# Fan infomation
maxFanDraw = 36.0       # 36W, based on 2.8W power consumption and 12 fans on the TBRe car
fanVoltage = 12.0       # 12V DC
                        # Fan current is the power divided by the voltage
fanCurrent = maxFanDraw / fanVoltage
coolingFloor = 285      # Min temperature the fans can physically get the batteries, 12oC

# Battery infomation
currDraw = 240.0        # 240A, continous current draw of 30A with 8 cells in parallel
battCapacity = 18.7     # 180Ah total capcity of the battery, note unit of hours!
battVoltageNom = 388.0  # 388V nominal voltage
battResistance = 0.2304 # 0.2304ohm, 108 in series, 6 in parallel, 12.8E-3 per cell
regenEfficiency = 0.55  # 55% Efficiecny of the velocity converted to regen power, and this power used for recharging
maxAllowTemp = 307           # Kelvin, 35oC
minAllowTemp = 288           # Kevlin, 15oC

# Define all the values to normalise between
# As the ANN needs normalised values
minSteering = -1.0
maxSteering = 1.0
minBreak = 0.0
maxBreak = 1.0
minThrottle = 0.0
maxThrottle = 1.0
minSpeedMPH = 0.0
maxSpeedMPH = 81.0
minBattTemp = 5.0 + 273               # These are wider than the fail criteria
maxBattTemp = 40.0 + 273
minAccel = 0.0
maxAccel = 183.0
minGenHeat = 0.0
maxGenHeat = 7600.0
minDt = 0.05
maxDt = 1.0
minFanSpeed = 0.0
maxFanSpeed = 1.0
maxBattCap = battCapacity
minBattCap = 0.0

# The temperature we want to aim for
aimTemperature = float(30 + 273)

# To halt the script
import sys
# Numpy matrices
import numpy as np
#Used to get the epoch time to calculate the time difference dt
import datetime
import time
#Regular expressions
import re
# Generate random numbers for the epsilon greedy 
import random
# Math functions
import math
# Allows deep copies to be made
import copy

# Setup the machine learning
# There are 8 states per tick, 4 ticks make up the agent
# SO the input shape is 32 (this can be scaled)
# Output is 6 different optinos, set fan to
# [0, 0.2, 0.4, 0.6, 0.8 1.0]
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.models import model_from_json

dataCount = 9       # Number of data values to learn from per row
dataDepth = 4       # The total number of rows per argent, oldest is popped off
outputDepth = 6     

model = Sequential()
model.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=((dataCount * dataDepth), )))
model.add(Activation('relu'))
# Hidden layer
model.add(Dense(150, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))
# Output layer, use linear so they're real world values
model.add(Dense(6, kernel_initializer='lecun_uniform'))
model.add(Activation('linear'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

# Functions

#Calculated the time difference in milliseconds
#Regex on each filepath to get the time variables
#Create an object to get the epoch time
#Differene in epoch time is the dt (in seconds)
def calculateDt(prev, current):
    #Calculate the epoch of the prev tick 
    #File last value is millisconds so convert to microseconds in datetime constructor
    p = re.search(r'.+center_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+).jpg', prev)
    prevTime = datetime.datetime(int(p.groups(1)[0]), int(p.groups(1)[1]), int(p.groups(1)[2]),
            int(p.groups(1)[3]), int(p.groups(1)[4]), int(p.groups(1)[5]), (int(p.groups(1)[6]) * 1000))
    prevEpoch = time.mktime(prevTime.timetuple())+(prevTime.microsecond/1000000.)

    #Calculate the epoch of the current tick 
    #File last value is millisconds so convert to microseconds in datetime constructor
    c = re.search(r'.+center_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+).jpg', current)
    currTime = datetime.datetime(int(c.groups(1)[0]), int(c.groups(1)[1]), int(c.groups(1)[2]),
            int(c.groups(1)[3]), int(c.groups(1)[4]), int(c.groups(1)[5]), (int(c.groups(1)[6]) * 1000))
    currEpoch = time.mktime(currTime.timetuple())+(currTime.microsecond/1000000.)

    dt = (currEpoch - prevEpoch)

    if dt > 1:
        if False:
            print("Warning : dt over 1, so setting to 1, from pictures:")
            print(prev)
            print(current)
        dt = 1

    return dt


# Sets up the state matrix (but not as a numpy matrix)
# Takes in the first amount of lines as to fill the datadepth
# We want in each data row:
#   dt, break, throttle, speedMPH, temp, acceleration, generated heat, the fan speed, battcapacity (mAh)
def initState(initialLines):
    state = list()

    prevLine = initialLines.pop(0).split(',')
    for line in initialLines:
        # Do all the processing for these lines, but no fan on and temp the same
        currLine = line.split(',')

        dt = calculateDt(prevLine[0], currLine[0])

        # Get all the normalised data
        normDT = (dt - minDt) / (maxDt - minDt)
        normBreak = (float(currLine[4]) - minBreak) / (maxBreak - minBreak)
        normThrot = (float(currLine[5]) - minThrottle) / (maxThrottle - minThrottle)
        normSpeedMPH = (float(currLine[6]) - minSpeedMPH) / (maxSpeedMPH - minSpeedMPH)
        normTemp = (float(Tamb) - minBattTemp) / (maxBattTemp  - minBattTemp)
        normAccel = (float(currLine[9]) - minAccel) / (maxAccel - minAccel)
        normHeat = (float(currLine[12]) - minGenHeat) / (maxGenHeat - minGenHeat)
        normFanSpeed = 0.0
        normBattCap = 1.0


        # Pack it all in the list
        dataRow = list()

        dataRow.append(normDT)
        dataRow.append(normBreak)
        dataRow.append(normThrot)
        dataRow.append(normSpeedMPH)
        dataRow.append(normTemp)
        dataRow.append(normAccel)
        dataRow.append(normHeat)
        dataRow.append(normFanSpeed)
        dataRow.append(normBattCap)

        state.append(dataRow)

        prevLine = currLine

    return state

# Take an action to change the fan speed
# Fan speed can only be changed up to 20% per dt, as per the main script
def actionFanSpeed(actionIndex, dt, coolingThrottle):
    fanSettings = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    setThrottle = fanSettings[actionIndex]

    if setThrottle == coolingThrottle:
        #Dont need to calculate anything
        return setThrottle
    elif setThrottle > coolingThrottle:
        #Need to calculate an increase
        #Increase is not able to be more than 10% per 0.2 seconds
        #Caulculate the maximum increase for the dt
        #Get the current x value (time)
        t = coolingThrottle * 2.0
        #Get the highest speed possible with the dt
        x = t + dt
        maxThrottle = x / 2.0
        #If the attempted fan speed to set is more than that maximum allowable for the dt
        #Just set it to the maximum allowable
        newThrottle = maxThrottle if (maxThrottle < setThrottle) else setThrottle
    else:
        #Need to calculate a decrease
        #Not able to decrease more than 10% every 1/3 seconds
        t = coolingThrottle * 2.0
        x = t - dt
        minThrottle = x / 2.0
        newThrottle = minThrottle if (minThrottle > setThrottle) else setThrottle

    return newThrottle

#Calculate the cooling which occurs, depending on the vehicle speed
#In m/s, and the accumulator 
#Uses fluid dynamics of dry air, with flow across a tube bank
def calculateCooling(coolingThrottle, dt):
    #Make sure the cooling throttle is between 0 and 1
    coolingThrottle = 1 if (coolingThrottle > 1) else coolingThrottle
    coolingThrottle = 0 if (coolingThrottle < 0) else coolingThrottle
    #Work out the air speed, max 10m/s
    coolAirSpeed = float(coolingThrottle) * 4.0

    #Caulcate Vmax, the max velocity of the air through the staggered tube bank arrangement
    St = 25.0 * (10.0 ** -3.0) 
    D = 18.65 * (10.0 ** -3.0) 
    Vmax = (St * coolAirSpeed) / (St - D)

    #Properties of dry air
    density = 1.177         
    dynamVis = 1.846 * (10.0 ** -5.0)
    criticalDimension = 0.01865

    #Calculate value of Re, pre computed values for desnity, cell size and dynamic viscosity
    #dt cancells out, but in the equations for clairty
    Re = (density * criticalDimension) / (dynamVis) * Vmax

    #Calculate the prandtl number, which is based on the Tfilm temperature 
    #Tfilm is the average of the Tamb and the heat of the object
    Pr = 0.706
    Prs = 0.708

    #Calculate the nusselt number, using the Re and the prandtl number
    #St and Sl are the dimensions of the staggered battery arangement
    St = 24.0 * (10.0 ** -3)
    Sl = 22.0 * (10.0 ** -3)
    Nu = (0.35 * ((St / Sl) ** (0.2)) * (Re ** 0.6) * (Pr ** 0.36) * ((Pr / Prs) ** 0.25))
        
    #Calculate the value of h, 
    h = ((2.624 * (10.0 ** -2.0) * Nu) / 0.01865)

    #Calculate the heat loss Q

    Q = h * 0.0044 * (maxAllowTemp - Tamb)

    #6 sub packs, 108 in series and 6 in parallel 636 total. However, with the depth the air cooling tends towards
    # to decrease, so only have the 6 sub packs with the 6 in the string being cooled in the model
    cellCount = 6.0 * 6.0

    #Calculate the cooling value over the cells, but multiplied by dt, and the number of sub packs
    coolingValue = Q * dt * cellCount

    return coolingValue

# Calcualte the drain on the batteriries
# Drain from running the fan
# Drain is very large approximation with a linear drain on the batteries!!
def calculateBatteryDrain(thisBattCapacity, coolingThrottle, drivingPower, dt):
    #Use the fan throttle to scale the instanteous current draw of the fans
    cellCurrentDraw = fanCurrent * coolingThrottle
    #Current is the flow per second, so also need to scale by dt
    cellCurrentDraw *= dt
    # Add the current that is drawn by the motor, by dividing the driving power by the nominal battery voltage
    # This power is already instantaneous in dt
    cellCurrentDraw += drivingPower / battVoltageNom

    if False:
        print("Instantaneous current " + str(cellCurrentDraw) + "A")
        print("dt : " + str(dt) + "s")

    #No power is consumed if the cellCurrentDraw is 0
    if cellCurrentDraw == 0:
        return thisBattCapacity

    #Remove this consumed power from the battery capacity
    #Work out how many seconds are left when this draw happens
    timeLeftHours = thisBattCapacity / cellCurrentDraw
    timeLeftSeconds = timeLeftHours * 3600.0
    #We are working in time units of dt, we are drawing for dt seconds
    timeLeftSeconds -= dt
    #Work in reverse to get the capacity after drawing the power
    timeLeftHours = timeLeftSeconds / 3600.0
    thisBattCapacity = timeLeftHours * cellCurrentDraw


    return thisBattCapacity

# Calculates the next state: takes in the current state, the new fan speed, 
# battery temp, current line, and the next data line
# Returns the next state, as well as raw values for temperature as this is used to calculate the reward
def advanceState(state, fanSpeed, temp, thisBattCapacity , currLine, nextLine):

    newState = state

    # Remove the oldest row from the state
    newState.pop(0)
    # First, work out the dt
    dt = calculateDt(currLine[0], nextLine[0])
    # Now need to calculate the cooling power
    coolingPower = calculateCooling(fanSpeed, dt)

    # Estimate the new battery life by changing the charge, ignore drain from the drive
    nextBattCap = calculateBatteryDrain(thisBattCapacity, fanSpeed, 0.0, dt)

    # Get the generated heat, the 12 index of the next line
    generatedHeat = float(nextLine[12])
    #The instantaneous power generated in the cells is the driving power minues the cooling power
    instantaneousPower = abs(generatedHeat) - coolingPower
    #Temperature cannot go lower than coolingFloor
    instantaneousPower = 0 if (temp < coolingFloor and instantaneousPower < 0) else instantaneousPower

    #Using q=m*c*Delta 
    dTemp = float(instantaneousPower) / (float(battMass) * float(specHeatCap) * dt)

    # Change the current temperature by the temperature change calculated
    temp = temp + dTemp
    #print("Temperature : " + str(temp - 273))

    # So now build the next data row (with normalising the data)
    # And then append it onto the state
    # Get all the normalised data
    normDT = (dt - minDt) / (maxDt - minDt)
    normBreak = (float(nextLine[4]) - minBreak) / (maxBreak - minBreak)
    normThrot = (float(nextLine[5]) - minThrottle) / (maxThrottle - minThrottle)
    normSpeedMPH = (float(nextLine[6]) - minSpeedMPH) / (maxSpeedMPH - minSpeedMPH)
    normTemp = (float(temp) - minBattTemp) / (maxBattTemp  - minBattTemp)
    normAccel = (float(nextLine[9]) - minAccel) / (maxAccel - minAccel)
    normHeat = (float(generatedHeat) - minGenHeat) / (maxGenHeat - minGenHeat)
    normFanSpeed = (float(fanSpeed) - minFanSpeed) / (maxFanSpeed - minFanSpeed)
    normBattCap = (nextBattCap - minBattCap) / (maxBattCap - minBattCap)

    # Pack it all in the list
    dataRow = list()

    dataRow.append(normDT)
    dataRow.append(normBreak)
    dataRow.append(normThrot)
    dataRow.append(normSpeedMPH)
    dataRow.append(normTemp)
    dataRow.append(normAccel)
    dataRow.append(normHeat)
    dataRow.append(normFanSpeed)
    dataRow.append(normBattCap)

    newState.append(dataRow)

    return newState, temp, nextBattCap


# Calculate the reward for taking the action - this is the reward that is used in refiencforcement
# It bakpropagates up through the ANN
# Takse in the state, and the battTemp (to make sure the temperature is where we want it)
# and the fanSpeed (want to use as little as possible to save battery life)
def calculateReward(battTemp, newBattTemp, thisBattCapacity, newBattCap):

    #print('--------------')
    #print("battTemp : " + str(battTemp))
    #print("newBattTemp : " + str(newBattTemp))
    #print("thisBattCapacity : " + str(thisBattCapacity))
    #print("newBattCap : " + str(newBattCap))

    reward = 0

    # First penalty is for consumping power 
    # Percentage difference 
    perDiff = abs(thisBattCapacity - newBattCap) / ((thisBattCapacity + newBattCap) / 2.0) * 100.0
    # Muiltiply by -100000 for the penality (so penality is about - single digits)
    reward -= perDiff * 100000.0 * 0.5

    #print("Battery drain : " + str(perDiff))
    #print("Drain reward : " + str(reward))

    #saveReward = copy.deepcopy(reward)

    # Penality or reward for moving away/towards target temperature
    if battTemp < newBattTemp:
        # moving upwards
        if newBattTemp < aimTemperature:
            reward += 2
            #print("Good movement")
        else:
            reward -= 2
            #print("Bad movement")
    else:
        # moving downwards
        if newBattTemp < aimTemperature:
            reward -= 2
            #print("Bad movement")
        else:
            reward += 2
            #print("Good movement")

    #print("movement reward : " + str(reward - saveReward))
    #saveReward = copy.deepcopy(reward)


    # Bell curve reward for being near the target temperature
    sigma = 0.4             # Standard deviation
    mean = aimTemperature   # Mean value
    multiplier = 8.0        # Multipy the standard distribution to give better ANN teachings

    reward += (1.0 / (sigma * (2.0 ** math.pi) ** 0.5)) * \
            math.exp(-1.0 * ((newBattTemp - mean) ** 2.0) / (2 * sigma ** 2.0)) * multiplier
    #print("temperature reward : " + str(reward - saveReward))

    # Big penaluty for going outside of the battery temperature
    if newBattTemp > maxAllowTemp or newBattTemp < minAllowTemp:
        reward -= 10

    #print(reward)

    return reward

#Process the input file to work out the power at each point
#Print it to a file
def processFile(filePath):
    #open this new file and get all the lines
    with open(filePath) as f:
        processLine = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        processLine = [x.strip() for x in processLine] 

    # Bin off the first line as it just contains all the titles
    processLine.pop(0)

    # Set up learning variables
    epochs = 1000                   # Run the learn 1000 times per simulation
    epsilon = 1                     # Using epsilon greedy, decreases over time
    gamma = 0.2                     # Used in the Q learning algorithm equation

    print(datetime.datetime.now().strftime("%a, %d %B %Y %I:%M:%S"))

    for i in range(epochs):
        # Set up the training run
        readLines = copy.deepcopy(processLine)
        # First thing to do is initialise the state
        # Keep the state as dynamic arrays so can be popped and appended
        state = initState(readLines[0:(dataDepth+1)])

        # Trim the lines that were sent to the state creation
        del readLines[0:dataDepth]
        prevLine = readLines.pop(0).split(',')

        battTemp = copy.deepcopy(Tamb)
        battCharge = 100.0
        fanSpeed = 0.0
        isRunning = True
        thisBattCapacity = copy.deepcopy(battCapacity)

        while isRunning:
            # Run the simulation
            currLine = readLines.pop(0).split(',')

            # Work out the dt
            dt = calculateDt(prevLine[0], currLine[0])

            # Get the quality values
            qVal = model.predict(np.array(state).reshape(1,(dataCount * dataDepth)), batch_size=1)

            # Choose a move with the epsilon greedy policy
            if random.random() < epsilon:
                # Make a random move
                actionIndex = np.random.randint(0, outputDepth)
            else:
                # Make the best action from the Quality function
                actionIndex = np.argmax(qVal)

            # Now need to take the action so we can observe Q'
            # First we need to work out the new fan speed with this taken action
            nextFanSpeed = actionFanSpeed(actionIndex, dt, fanSpeed)

            # We recieve back the battery temperature again seperately as we want a non-normalised copy of it
            nextState, nextBattTemp, nextBattCap = \
                    advanceState(state, nextFanSpeed, battTemp, thisBattCapacity, currLine, readLines[0].split(','))

            # Next we need to observe the reward from taking this action
            # The rewards are worked out by the changes in temperature and remaining capavity
            # Raw values are passed so the values do not need to be un-normalised from the state matrix
            reward = calculateReward(battTemp, nextBattTemp, thisBattCapacity, nextBattCap) 

            # Get the max Q'(s', a)
            newQ = model.predict(np.array(nextState).reshape(1,(dataCount * dataDepth)), batch_size=1)
            maxQ = np.max(newQ)

            y = np.zeros((1, outputDepth))
            y[:] = qVal[:]

            if len(readLines) == 1:
                # Got to the end of the simulation
                # WE cant do the next state as it requires a state after to run
                isRunning = False
                update = reward
            else:
                # Carry on going
                prevLine = currLine
                update = (reward + gamma * maxQ)

            # Set the output to what we want it to be, and back propogate through the network
            y[0][actionIndex] = update

            model.fit(np.array(state).reshape(1, (dataCount * dataDepth)), y, batch_size=1, nb_epoch=1, verbose=0)

            # We are now on the next state
            state = nextState
            battTemp = nextBattTemp
            fanSpeed = nextFanSpeed
            thisBattCapacity = nextBattCap

            # Game ends if the battery temperature went out of bounds!
            if battTemp > maxAllowTemp or battTemp < minAllowTemp:
                isRunning = False
        # End game loop

        # Decrease epsilon over time
        # This gets the system to increasingly rely on its knowledge over the random choice
        if epsilon > 0.1:
            epsilon -= (1/epochs)

        # Every 10 % of the way print the current time, and the percentage
        if ((i + 1) % (epochs / 10.0)) == 0:
            print("---------------------------------------------")
            print(str(((i + 1) / epochs) * 100) + "% completed...")
            print(datetime.datetime.now().strftime("%a, %d %B %Y %I:%M:%S"))


        # Print some things at the end of each game to see how close we are to targets :)
        print("---------------------------------------------")
        print("Completed game " + str(i + 1) + "/" + str(epochs))
        print("End temperature : " + str(battTemp))
        batteryPercentage = (thisBattCapacity / battCapacity) * 100.0
        #Get this as a string with two decimal places (cut, not rounded)
        wholePercentage = str(batteryPercentage).split('.')[0]
        decimPercentage = str(batteryPercentage).split('.')[1]
        batteryRemaining = str(wholePercentage + "." + decimPercentage[0:2])
        print("End capacity : " + str(batteryRemaining))
    
    # Simulation finished, print model to JSON file
    print("---------------------------------------------")
    print(datetime.datetime.now().strftime("%a, %d %B %Y %I:%M:%S"))
    modelJSON = model.to_json()
    with open("qrienforcmentmodel.json", "w") as JSONFile:
        JSONFile.write(modelJSON)
    # Serialise weights to HDF5
    model.save_weights("qrienforcmentmodel.h5")
    print("Model saved to disk...")



# File with the list of files to process
masterFile = "/Users/Matt/google_drive/documents/uni/year_3/EE30147_40148_GDBP_MEng/individual/scripts/traindatatoprocess"

#Open the list of files to go through, 
try:
    with open(masterFile) as f:
        processFilePaths = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        processFilePaths = [x.strip() for x in processFilePaths] 
except IOError:
    print("Error, unable to find list file : " + filePath)

#Remove first item from processFilePaths as this is just the file header
processFilePaths.pop(0)

#For every required file
for row in processFilePaths:
    #Try to process this file
    try:
        print("Processing file : " + row)
        processFile(row)
    except IOError:
        print("Error, unable to find list file : " + row)
