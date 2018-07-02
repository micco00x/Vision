import argparse
import natsort
import numpy as np
import cv2
import os

IMAGE_SIZE = 64

# Parse arguments from command line:
parser = argparse.ArgumentParser(description="Convert a dataset to .npz")
parser.add_argument("--dataset", required=True, help="Dataset to convert in .npz")
args = parser.parse_args()

# Dataset (X, y):
videos = []
videos_y = []

# Classes of the videos:
video_classes = set()

with open(args.dataset) as dataset_file:
    video_folders = dataset_file.readlines()
    for idx, video_folder in enumerate(video_folders):

        print("Progress: {:2.1%}".format(idx / len(video_folders)), end="\r")

        # Remove \n from the video folder path:
        video_folder = video_folder.rstrip("\n")

        video = []

        # Save video class for later:
        video_class = video_folder[:video_folder.rfind("/")]
        video_classes.add(video_class)
        videos_y.append(video_class)

        # For each video folder (naturally sorted) extract the frames:
        for video_frame in natsort.natsorted(os.listdir(video_folder)):
            frame = cv2.imread(os.path.join(video_folder, video_frame))
            frame = cv2.resize(frame, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            video.append(frame)

        videos.append(video)
    print("Progress: COMPLETED")

    # Convert videos to np.array and classes to indices:
    videos = np.array(videos)
    video_classes = list(video_classes)
    #videos_y = get_one_hot_encoding(videos_y, list(video_classes))
    videos_y = np.array([video_classes.index(elem) for elem in videos_y], dtype=np.uint8)

    print("Shape of the np.array(s):")
    print(videos.shape)
    print(videos_y.shape)

    idx_extension = args.dataset.rfind(".")
    np.savez(args.dataset[:idx_extension] + "_frames.npz", videos=videos, videos_y=videos_y)
