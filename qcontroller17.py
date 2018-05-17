class qcontroller17:
    'A few changes made. Data sampled with a 6 place moving average, and dt removed from the q=mc^t equation. Uisng ninary crossentropy again'

    def __init__(self):
        # Define all the values to normalise between
        # As the ANN needs normalised values
        self.minSteering = -1.0
        self.maxSteering = 1.0
        self.minBreak = 0.0
        self.maxBreak = 1.0
        self.minThrottle = 0.0
        self.maxThrottle = 1.0
        self.minSpeedMPH = 0.0
        self.maxSpeedMPH = 81.0
        self.minBattTemp = 5.0 + 273        # These are wider than the fail criteria
        self.maxBattTemp = 40.0 + 273
        self.minAccel = 0.0
        self.maxAccel = 183.0
        self.minGenHeat = 0.0
        self.maxGenHeat = 7600.0
        self.minDt = 0.05
        self.maxDt = 1.0
        self.minFanSpeed = 0.0
        self.maxFanSpeed = 1.0
        self.maxBattCap = 18.7
        self.minBattCap = 0.0

        # Load the ml packages
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.optimizers import RMSprop
        from keras.models import model_from_json
        from keras.callbacks import Callback
        # Need numpy arrays
        import numpy as np
        self.np = np

        rms = RMSprop()

        self.dataCount = 9       # Number of data values to learn from per row
        self.dataDepth = 4       # The total number of rows per argent, oldest is popped off
        self.outputDepth = 6     

        # Load our created model!
        jsonFile = open('ml_attempt_17/qrienforcmentmodel.json', 'r')
        loadedModelJSON = jsonFile.read()
        jsonFile.close()
        self.model = model_from_json(loadedModelJSON)
        # Load the weights into thr new model
        self.model.load_weights("ml_attempt_17/qrienforcmentmodel.h5")
        self.model.compile(loss='binary_crossentropy', optimizer=rms)
        print("Loaded model from disk...")

        # Finally, create the state of the system
        innerList = [0] * self.dataCount
        state = [innerList] * self.dataDepth

        self.state = state
        
    # Take an action to change the fan speed
    # Fan speed can only be changed up to 20% per dt, as per the main script
    def __actionFanSpeed(self, actionIndex, dt, coolingThrottle):
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

    #THis is the same function stucture that every controller requires
    #Implemented differently by every controller
    #Returns a new fan value
    def runController(self, lineData, dt, acceleration, temperature, generatedHeat, fanSpeed, remainCapacity):
        # So now build the next data row (with normalising the data)
        # And then append it onto the state
        # Get all the normalised data
        normDT = (dt - self.minDt) / (self.maxDt - self.minDt)
        normBreak = (float(lineData[4]) - self.minBreak) / (self.maxBreak - self.minBreak)
        normThrot = (float(lineData[5]) - self.minThrottle) / (self.maxThrottle - self.minThrottle)
        normSpeedMPH = (float(lineData[6]) - self.minSpeedMPH) / (self.maxSpeedMPH - self.minSpeedMPH)
        normTemp = (float(temperature) - self.minBattTemp) / (self.maxBattTemp  - self.minBattTemp)
        normAccel = (float(acceleration) - self.minAccel) / (self.maxAccel - self.minAccel)
        normHeat = (float(generatedHeat) - self.minGenHeat) / (self.maxGenHeat - self.minGenHeat)
        normFanSpeed = (float(fanSpeed) - self.minFanSpeed) / (self.maxFanSpeed - self.minFanSpeed)
        normBattCap = (remainCapacity - self.minBattCap) / (self.maxBattCap - self.minBattCap)

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

        self.state.append(dataRow)
        self.state.pop(0)

        #print(self.state)

        # Make the control move
        qVal = self.model.predict(self.np.array(self.state).reshape(1,(self.dataCount * self.dataDepth)), batch_size=1)
        print(qVal)
        actionIndex = self.np.argmax(qVal)
        throttleValue = self.__actionFanSpeed(actionIndex, dt, fanSpeed)

        print(throttleValue)

        return throttleValue

        
