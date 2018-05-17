# My script to process the generated data
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

# GA variables
generationCount = 5
populationCount = 50
mutationRate = 0.01

# Some variables to do with the traniing of the controller
reqSigFig = 4           # Number of signiicant figures we need to variables to
maxkp = 2.0
maxki = 2.0
maxkd = 2.0

# Chromosomes length
chromKPlen = math.ceil(math.log(((maxkp - 0) * 10 ** reqSigFig), 2))
chromKIlen = math.ceil(math.log(((maxki - 0) * 10 ** reqSigFig), 2))
chromKDlen = math.ceil(math.log(((maxkd - 0) * 10 ** reqSigFig), 2))

# Generate a random binary string of input length
def generateBinaryString(chromosomeLength):
    returnString = ''

    for i in range(1, chromosomeLength):
        if random.random() < 0.5:
            returnString += '0'
        else:
            returnString += '1'

    return returnString

# Decode chromosome
# Decodes the chromoesome string to the float digit
def decodeChromosome(chromosome, chromoesomeMaxVal):
    decode = 0.0 + float(int(chromosome, 2)) * (float(chromoesomeMaxVal) - 0.0) / (2.0 ** len(chromosome) - 1)
    return decode



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

class pidcontroller:
    'This is a pid controller '

    def __init__(self, kp, ki, kd):
        # Temperature setpoint for the contoller
        self.aimTemperature = 273.0 + 30

        # Gains built from the ML model
        self.kp = kp
        self.ki = ki
        self.kd = kd

        print("Gains loaded for the PID controller:- kp : " + str(self.kp)
            + ", ki : " + str(self.ki) + ", kd : " + str(self.kd))

        # Values to normalise the nominal controller value between
        self.contLow = -5.0
        self.contHig = 5.0

        # Initial state of the intergrator
        self.uiPrev = 1.0
        # Previous error value
        self.ePrev = 1.0

    # Take an action to change the fan speed
    # Fan speed can only be changed up to 20% per dt, as per the main script
    def __actionFanSpeed(self, setThrottle, dt, coolingThrottle):

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
        
    #THis is the same function stucture that every controller requires
    #Implemented differently by every controller
    #Returns a new fan value
    def runController(self, dt, fanSpeed, temperature):
        # Implement PID
        # Error values are swapped as we want the controller on when the temperature is higher
        e = temperature - self.aimTemperature
        print('-----------------------')
        print(temperature)
        print(e)

        ui = self.uiPrev + (1/self.ki) * dt * e
        ud = (1/self.kd) + (e - self.ePrev) / dt

        # Save the previous values
        self.uiPrev = ui
        self.ePrev = e

        # Generate the control effort
        u = self.kp * e  + ud

        print(u)

        un = 2 * (u - self.contLow) / (self.contHig - self.contLow) - 1

        throttleValue = un if un >= 0 else 0

        print(throttleValue)

        return throttleValue

class geneticLearner:
    'An individual in the society'

    def __init__(self, chromKP, chromKI, chromKD, initialTemp):
        self.chromKP = chromKP
        self.chromKI = chromKI
        self.chromKD = chromKD

        self.temperature = copy.deepcopy(initialTemp)

        KP = decodeChromosome(chromKP, maxkp)
        KI = decodeChromosome(chromKI, maxki)
        KD = decodeChromosome(chromKD, maxkd)

        self.controller = pidcontroller(KP, KI, KD)

        # Setup some values such as the fan speed
        self.fanspeed = 0.0

        # Set to true of the temperature every goes out of bounds
        self.failStatus = False

        # Just need to track the charge just to display it on a chart
        self.thisBattCap = copy.deepcopy(battCapacity)

        # Finally, reset the fitness value
        self.fitness = 0.0
        self.fitnessBounds = 1.0        #+- 1 Kelvin
        self.beginTracking = False      # Begins false as we start far away from the temperature

    # Runs the genetic controller, just the normal kind of control things taking place
    # Keep track of fitness though
    def runController(self, dt, generatedHeat):
        newThrottle = self.controller.runController(dt, self.fanspeed, self.temperature)

        #This is the number of watts of heat taken away per dt from the airflow
        coolingPower = calculateCooling(coolingThrottle, dt)

        #The instantaneous power generated in the cells is the driving power minues the cooling power
        instantaneousPower = abs(generatedHeat) - coolingPower
        #Temperature cannot go lower than coolingFloor
        instantaneousPower = 0 if (temp < coolingFloor and instantaneousPower < 0) else instantaneousPower

        #Using q=m*c*Delta 
        dTemp = float(instantaneousPower) / (float(battMass) * float(specHeatCap) * dt)

        # Change the current temperature by the temperature change calculated
        self.temperature += dTemp

        self.thisBattCapacity = calculateBatteryDrain(dt, newThrottle, 0.0, self.thisBattCapacity)

        if self.temp < minAllowTemp or temp > maxAllowTemp:
            print("Failure, battery temperature at " + str(temp - 273))
            self.fitness = 0.0
            self.failStatus = True
        else:
            # Carry on looping though all the data
            if self.beginTracking:
                if (self.temperature <= self.aimTemperature + self.fitnessBounds and 
                        self.temperature >= self.aimTemperature - self.fitnessBounds):
                            print("Hello")



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
    epochs = generationCount                   # Run the learn 1000 times per simulation


    print(datetime.datetime.now().strftime("%a, %d %B %Y %I:%M:%S"))

    # Initialise the generation
    generationGroup = []
    for count in range(populationCount):
        chromKP = generateBinaryString(chromKPlen)
        chromKI = generateBinaryString(chromKPlen)
        chromKD = generateBinaryString(chromKPlen)
        cont = geneticLearner(chromKP, chromKI, chromKD, Tamb)
        generationGroup.append(cont)


    for i in range(epochs):
        # Set up the training run
        readLines = copy.deepcopy(processLine)
        # First thing to do is initialise the state
        # Keep the state as dynamic arrays so can be popped and appended

        # Trim the lines that were sent to the state creation

        battTemp = copy.deepcopy(Tamb)
        battCharge = 100.0
        fanSpeed = 0.0
        isRunning = True
        thisBattCapacity = copy.deepcopy(battCapacity)

        lineCounter = 0

        prevLine = readLines.pop(0).split(',')

        while isRunning:
            # Run the simulation
            currLine = readLines.pop(0).split(',')

            # Work out the dt
            dt = calculateDt(prevLine[0], currLine[0])

            if len(readLines) > 1:
                prevLine = currLine
            else:
                isRunning = False


        # Every 10 % of the way print the current time, and the percentage
        if ((i + 1) % (epochs / 10.0)) == 0:
            print("---------------------------------------------")
            print(str(((i + 1) / epochs) * 100) + "% completed...")
            print(datetime.datetime.now().strftime("%a, %d %B %Y %I:%M:%S"))


        # Print some things at the end of each game to see how close we are to targets :)
        print("---------------------------------------------")
        print("Completed game " + str(i + 1) + "/" + str(epochs))
        #print("End temperature : " + str(battTemp))
        #batteryPercentage = (thisBattCapacity / battCapacity) * 100.0
        ##Get this as a string with two decimal places (cut, not rounded)
        #wholePercentage = str(batteryPercentage).split('.')[0]
        #decimPercentage = str(batteryPercentage).split('.')[1]
        #batteryRemaining = str(wholePercentage + "." + decimPercentage[0:2])
        #print("End capacity : " + str(batteryRemaining))
    
    # Simulation finished, print model to JSON file
    print("---------------------------------------------")
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
