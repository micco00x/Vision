import os
import glob
import cv2
import shutil

numberFrames = 40
trim = 0.2
videoFolder = "Videos"
framesFolder = "Frames"

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

    for video in files:

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