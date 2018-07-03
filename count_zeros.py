import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Count how many frames are zeros")
parser.add_argument("--npz", help=".npz file")
args = parser.parse_args()

npzfile = np.load(args.npz)
X = npzfile["videos"]
y = npzfile["videos_y"]

cnt = 0

for video_idx in range(X.shape[0]):
    print("Progress: {:2.3}".format(video_idx / X.shape[0]), end="\r")
    for frame_idx in range(X.shape[1]):
        if np.any(X[video_idx,frame_idx,:,:,:]):
            cnt = cnt + 1
print("Progress: COMPLETE")

print("\n{}/{} ({})".format(cnt, X.shape[0] * X.shape[1], cnt/(X.shape[0] * X.shape[1])))
