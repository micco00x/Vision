import os
import sys
import random
from sklearn.model_selection import train_test_split
import pickle
import argparse

# Parse arguments from command line:
parser = argparse.ArgumentParser(description="Extract frames from a set of videos and create a dataset")
parser.add_argument("--trainingsize", required=False, default=0.8, help="Size of the training set (default=0.8)")
parser.add_argument("--framesfolder", required=False, default="Frames", help="Folder that will contain frames (default=Frames)")
args = parser.parse_args()

framesFolder = args.framesfolder
filesList = []
trainingSize = args.trainingsize

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
train, test = train_test_split(filesList, train_size=trainingSize)

f = open(framesFolder + '/train.txt', 'w')
f.write("\n".join(train))
f.close()

f = open(framesFolder + '/test.txt', 'w')
f.write("\n".join(test))
f.close()
