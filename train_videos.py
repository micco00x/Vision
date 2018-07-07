import tensorflow as tf
import numpy as np
import argparse
import sys
import time
import os

import VideoClassifier

DEFAULT_LOGS_DIR = "logs"

if __name__ == "__main__":
    # Parse arguments from command line:
    parser = argparse.ArgumentParser(description="Train a network to classify videos in activities")
    parser.add_argument("--train", required=True, help="Train .npz file that stores tensors used to train the network")
    parser.add_argument("--test", required=True, help="Test .npz file that stores tensors used to train the network")
    parser.add_argument("--logs", required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help="Logs and checkpoints directory (default=logs)")
    parser.add_argument("--batch_size", default=32, type=int, help="Size of the batch for the LSTM (default=32)")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate for the RMSProp optimizer (default=1e-3)")
    parser.add_argument("--epochs1", default=10, type=int, help="Up to which epoch to train the LSTM for the first stage (default=10)")
    parser.add_argument("--epochs2", default=15, type=int, help="Up to which epoch to train the LSTM for the second stage (default=15)")
    parser.add_argument("--lstm_hidden", default=256, type=int, help="Size of the hidden layer for the LSTM")
    parser.add_argument("--save_checkpoint", default=10, type=int, help="Number of epochs before saving the model (default=10)")
    args = parser.parse_args()

    # Load tensors (train):
    npzfile = np.load(args.train)
    train_videos = npzfile["videos"]
    train_classes = npzfile["videos_y"]

    # Load tensors (test):
    npzfile = np.load(args.test)
    test_videos = npzfile["videos"]
    test_classes = npzfile["videos_y"]

    # videos should be [num videos, num frames, width, height, channels],
    # classes should be [num_videos, ]:
    print(train_videos.shape)
    print(train_classes.shape)
    print(test_videos.shape)
    print(test_classes.shape)

    # Convert videos' frames from uint8 to float32, specifically from [0,255] to [0,1]:
    train_videos = train_videos.astype(np.float32) / 255
    test_videos = test_videos.astype(np.float32) / 255
    #print("Checking videos are in range [0,1]:", (0 <= videos).all() and (videos <= 1).all())

    # LSTM hyperparameters:
    lstm_hidden = args.lstm_hidden
    num_classes = len(set(train_classes))
    learning_rate = args.learning_rate

    with tf.Session() as sess:
        T = time.strftime("%Y%m%dT%H%M")
        videoClassifier = VideoClassifier.VideoClassifier(lstm_hidden, num_classes,
                                                          [None] + list(train_videos.shape[1:]),
                                                          [None] + list(train_classes.shape[1:]))
        videoClassifier.set_tensorboard_folder(os.path.join(args.logs, "lstm" + T), sess)
        videoClassifier.set_checkpoint_folder(os.path.join("weights", "lstm" + T))
        videoClassifier.train(train_videos, train_classes, test_videos, test_classes,
                              tf.train.RMSPropOptimizer(learning_rate=args.learning_rate),
                              args.batch_size, args.epochs1, sess, save_checkpoint=args.save_checkpoint)
        videoClassifier.save(os.path.join(videoClassifier.checkpoint_folder, "first_stage.ckpt"), sess)
        videoClassifier.train(train_videos, train_classes, test_videos, test_classes,
                              tf.train.RMSPropOptimizer(learning_rate=args.learning_rate / 10),
                              args.batch_size, args.epochs2, sess, save_checkpoint=args.save_checkpoint)
        videoClassifier.save(os.path.join(videoClassifier.checkpoint_folder, "second_stage.ckpt"), sess)
