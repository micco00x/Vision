import os
import sys
import random
from sklearn.model_selection import train_test_split
import pickle

framesFolder = "Frames"
filesList = []
trainingSize = .8

# create folder Frames
if not os.path.exists(framesFolder):
    sys.exit("ERROR: Folder 'Frames' not found!")

# list of activities
folders = os.listdir(framesFolder)

for f in folders:
    # list of videos
    files = os.listdir(framesFolder + "/" + f)

    for file in files:
        filesList.append(framesFolder + "/" + f + "/" + file)

random.shuffle(filesList)
train, test = train_test_split(filesList, train_size = trainingSize)

f = open(framesFolder + '/train.txt', 'w')
f.write("\n".join(train))
f.close()

f = open(framesFolder + '/test.txt', 'w')
f.write("\n".join(test))
f.close()