import numpy as np
import PIL
import cv2
import argparse
import sys
import os
import natsort

sys.path.append("third_party/Mask_RCNN/") # To find local version of the library
from mrcnn import model as modellib
from mrcnn import visualize
import activity
import MaskExam

import matplotlib.pyplot as plt

#def get_one_hot_encoding(y, y_classes):
#    _y = np.zeros((len(y), len(y_classes)))
#    for idx, y_elem in enumerate(y):
#        _y[idx][y_classes.find(y_elem)] = 1
#    return _y

SHOW_IMAGES = False
DEFAULT_LOGS_DIR = "logs"
IMAGE_SIZE = 64

class ExtendedInferenceConfig(activity.ExtendedCocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9

# Parse arguments from command line:
parser = argparse.ArgumentParser(description="Train a network to classify videos in activities")
parser.add_argument("--dataset", required=True, help=".txt file with paths to videos' frames")
parser.add_argument('--logs', required=False,
                    default=DEFAULT_LOGS_DIR,
                    metavar="/path/to/logs/",
                    help='Logs and checkpoints directory (default=logs)')
parser.add_argument('--model',
                    metavar="/path/to/weights.h5",
                    help="Path to weights .h5 file")
args = parser.parse_args()

# Load the pretrained Mask-RCNN model:
config = ExtendedInferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

# Load the weights given the path:
model.load_weights(args.model, by_name=True)

# Generate colors (unique for each class):
colors = MaskExam.generate_class_colors()

video_classes = set()

# Dataset (X, y):
videos = []
videos_y = []

# Open the dataset with the list of path to video folders:
with open(args.dataset) as dataset_file:
    # Go through each video folder:
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
            # Read the image and convert it to np.uint8:
            #print("Reading " + os.path.join(video_folder, video_frame))
            frame = cv2.imread(os.path.join(video_folder, video_frame))
            #masked_image = frame.astype(np.uint8).copy()
            masked_image = np.zeros(frame.shape, dtype=np.uint8)

            # Get the masks of the frame and apply them onto the frame itself:
            results = model.detect([frame], verbose=0)
            r = results[0]
            masks = r["masks"]
            class_ids = r["class_ids"]

            for idx in range(masks.shape[2]):
                mask = masks[:,:,idx]
                color = colors[class_ids[idx]]
                masked_image = visualize.apply_mask(masked_image, mask, color, alpha=1.0)

            # Resize images:
            frame = cv2.resize(frame, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            masked_image = cv2.resize(masked_image, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            video.append(masked_image)

            # Plot images:
            if SHOW_IMAGES:
                original_img = PIL.Image.fromarray(frame, "RGB")
                generated_img = PIL.Image.fromarray(masked_image, "RGB")
                fig = plt.figure()
                fig.add_subplot(1, 2, 1)
                plt.imshow(original_img)
                fig.add_subplot(1, 2, 2)
                plt.imshow(generated_img)
                plt.show()

        videos.append(video)
    print("Progress: COMPLETED")

# Convert videos to np.array and classes to indices:
videos = np.array(videos)
video_classes = list(video_classes)
#videos_y = get_one_hot_encoding(videos_y, list(video_classes))
videos_y = np.array([video_classes.index(elem) for elem in videos_y])

# TODO: videos should be [num videos, num frames].
print(videos.shape)
print(videos_y.shape)
