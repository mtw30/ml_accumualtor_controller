# Script goes through the trainingdata set that is produced by the generateHeatProfile.py script
# Performs an x data moving average, where x is a variable I set

import sys

# dt is roughly 0.08 so this would very roughly every quater second
x = 6

#Process the input file to work out the power at each point
#Print it to a file
def processFile(filePath):
    #open this new file and get all the lines
    with open(filePath) as f:
        processLine = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        proceossLine = [x.strip() for x in processLine] 

    outputName = str(x) + "_avg_" + filePath.split('/')[-1]

    outF = open(outputName, "w")

    # Write the first to lines straight away as they are going to be exactly the same
    outF.write(processLine.pop(0))

    firstDataLine = processLine.pop(0)
    columnCount = len(firstDataLine.split(','))

    outF.write(firstDataLine)

    # Loop through each x sets of data to average them
    for run in range(0, (len(processLine) // x)):
        data = [0] * columnCount

        for i in range(0, x):
            theLine = processLine.pop(0).split(',')
            # The first three lines are the camera bits, so just use the last value
            for j in range(3, columnCount):
                data[j] += float(theLine[j])

            # One the sum has been collected, get the camera file
            outLine = ''
            outLine += theLine[0] + ','
            outLine += theLine[1] + ','
            outLine += theLine[2] + ','
            for j in range(3, columnCount):
                outLine += str(data[j] / x) + ","

        outLine = outLine.rstrip(',')

        outF.write(outLine + "\n")

    # Just bin off the last lines haha


# File with the list of files to process
# Uses the same masterFile as the qlearningscript.py
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
