import os
import glob
import cv2
import shutil
import argparse

# Parse arguments from command line:
parser = argparse.ArgumentParser(description="Extract frames from a set of videos and create a dataset")
parser.add_argument("--numframes", required=False, default=40, help="Number of frames to extract from each video (default=40)")
parser.add_argument("--trim", required=False, default=0.2, help="How much of the video to trim (default=0.2)")
parser.add_argument("--videofolder", required=False, default="Videos", help="Folder containing the videos (default=Videos)")
parser.add_argument("--framesfolder", required=False, default="Frames", help="Folder that will contain frames (default=Frames)")
args = parser.parse_args()

numberFrames = args.numframes
trim = args.trim
videoFolder = args.videofolder
framesFolder = args.framesfolder

# list of activities
folders = os.listdir(videoFolder)

# create folder Frames
if os.path.exists(framesFolder):
    shutil.rmtree(framesFolder)
os.makedirs(framesFolder)

for f in folders:
    # list of videos
    files = glob.glob(videoFolder + "/" + f + "/*.mp4")

    # create exercise folder
    if not os.path.exists(framesFolder + "/" + f):
        os.makedirs(framesFolder + "/" + f)

    for idx, video in enumerate(files):

        print("Reading {}: {:2.1%}".format(f, idx / len(files)), end="\r")

        # create video folder
        _, videoName = os.path.split(os.path.splitext(video)[0])
        if not os.path.exists(framesFolder + "/" + f + "/" + videoName):
            os.makedirs(framesFolder + "/" + f + "/" + videoName)

        # open video
        vidcap = cv2.VideoCapture(video)

        # video data
        frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)*(1-2*trim))

        for i in range(numberFrames):
            frame_number = int(frames/numberFrames*i+frames*trim)
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            success, image = vidcap.read()

            if not success:
                print("Error")

            cv2.imwrite(framesFolder + "/" + f + "/" + videoName + "/%d.jpg" % i, image)  # save frame as JPEG file

    print("Reading {}: COMPLETED".format(f))
