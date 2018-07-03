import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser(description="Resize the frames of a .npz dataset")
parser.add_argument("--npz", help=".npz file")
parser.add_argument("--size", type=int, help="New size of the frame")
args = parser.parse_args()

npzfile = np.load(args.npz)
X = npzfile["videos"]
videos_y = npzfile["videos_y"]

videos = []

cnt = 0

for video_idx in range(X.shape[0]):
    print("Progress: {:2.1%}".format(video_idx / X.shape[0]), end="\r")
    video = []
    for frame_idx in range(X.shape[1]):
        frame = X[video_idx,frame_idx,:,:,:]
        video.append(cv2.resize(frame, dsize=(args.size, args.size), interpolation=cv2.INTER_CUBIC))
    videos.append(video)
print("Progress: COMPLETE")

videos = np.array(videos)

print(videos.shape)
print(videos_y.shape)

idx_extension = args.npz.rfind(".")
np.savez(args.npz[:idx_extension] + "_" + str(args.size) + ".npz", videos=videos, videos_y=videos_y)
