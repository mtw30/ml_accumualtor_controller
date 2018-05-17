# This python script goes through the output.txt and puts the data in three columns in a csv file
# This is so the data can be plotted to see how long it takes for the learning to trend towards
# its final values

# used for regex
import re

gameEx = re.compile(r'Completed game (\d+)')
tempEx = re.compile(r'End temperature : (\d+\.\d*)')
chrgEx = re.compile(r'End capacity : (\d+\.\d*)')

gameNo = 0.0
temp = 0.0
cap = 0.0

outF = open("processTrainedOutput.csv", "w")

with open("output.txt") as f:
    for line in f:
        # Chomp the newline from the EOL
        processLine = line.rstrip()

        # Create the print line
        if gameEx.search(processLine):
            printLine = str(gameEx.search(processLine).groups(1)[0]) + ","

        if tempEx.search(processLine):
            printLine += str(tempEx.search(processLine).groups(1)[0]) + ","

        if chrgEx.search(processLine):
            printLine += str(chrgEx.search(processLine).groups(1)[0]) + "\n"
            outF.write(printLine)


