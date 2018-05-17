class pidcontroller0:
    'This is a pid controller '

    def __init__(self):
        # Temperature setpoint for the contoller
        self.aimTemperature = 273.0 + 30

        # Gains built from the ML model
        self.kp = 1.0
        self.ki = 1.0
        self.kd = 1.0

        print("Gains loaded for the PID controller:- kp : " + str(self.kp)
            + ", ki : " + str(self.ki) + ", kd : " + str(self.kd))

        # Values to normalise the nominal controller value between
        self.contLow = -5.0
        self.contHig = 5.0

        # Initial state of the intergrator
        self.uiPrev = 1.0
        # Previous error value
        self.ePrev = 1.0
        
    #THis is the same function stucture that every controller requires
    #Implemented differently by every controller
    #Returns a new fan value
    def runController(self, lineData, dt, acceleration, temperature, generatedHeat, fanSpeed, remainCapacity):
        # Implement PID
        # Error values are swapped as we want the controller on when the temperature is higher
        e = temperature - self.aimTemperature
        print('-----------------------')
        print(temperature)
        print(e)

        ui = self.uiPrev + (1/self.ki) * dt * e
        ud = (1/self.kd) * (e - self.ePrev) / dt

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

        
