class simplecontroller:
    'Simple fan controller implementing a relay with hysteresis'

    def __init__(self):
        #Contoller constructor, define the hysteresis values
        self.h_p = 31.0 + 273.0
        self.h_m = 29.0 + 273.0
        self.m_p = 1.0
        self.m_m = 0.0

    #THis is the same function stucture that every controller requires
    #Implemented differently by every controller
    #Returns a new fan value
    def runController(self, lineData, dt, acceleration, temperature, generatedHeat, coolingThrottle, thisBattCapacity):
        #This particular controller runs a relay with hysteresis, with values defined in the 
        #Contoller constructor
        throttleValue = 0.0
        
        if temperature < self.h_m:
            throttleValue = self.m_m
        elif temperature > self.h_p:
            throttleValue = self.m_p

        return throttleValue

        
