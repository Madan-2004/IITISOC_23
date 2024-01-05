def getGroupSize(channels):
    if channels>=32:
        goalSize=8
    else:
        goalSize=4
    if channels%goalSize==0:
        return goalSize