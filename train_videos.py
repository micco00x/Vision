import tensorflow as tf
import numpy as np
import argparse
import sys
import time

import VideoClassifier

import matplotlib.pyplot as plt
import itertools
import sklearn

DEFAULT_LOGS_DIR = "logs"
CLASS_NAMES = ["Doing crunches", "Doing step aerobics", "Elliptical trainer",
               "Kneeling", "Rope skipping", "Running a marathon", "Spinning",
               "Tumbling", "Using parallel bars","Using the balance beam", "Using the pommel horse",
               "Using the rowing machine", "Using uneven bars", "Zumba"]

# From: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    np.set_printoptions(precision=2)
    plt.figure(figsize=(7.5, 7), dpi=480, tight_layout=True)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show()
    plt.savefig("images/" + title)

if __name__ == "__main__":
    # Parse arguments from command line:
    parser = argparse.ArgumentParser(description="Train a network to classify videos in activities")
    parser.add_argument("--train", help="Train .npz file that stores tensors used to train the network")
    parser.add_argument("--test", help="Test .npz file that stores tensors used to train the network")
    parser.add_argument("--logs", required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help="Logs and checkpoints directory (default=logs)")
    parser.add_argument("--model", required="--dataset" in sys.argv,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument("--batch_size", default=32, type=int, help="Size of the batch for the LSTM (default=32)")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate for the RMSProp optimizer (default=1e-3)")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train the LSTM (default=10)")
    parser.add_argument("--lstm_hidden", default=256, type=int, help="Size of the hidden layer for the LSTM")
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
    epochs = args.epochs

    with tf.Session() as sess:
        T = time.strftime("%Y%m%dT%H%M")
        videoClassifier = VideoClassifier.VideoClassifier(lstm_hidden, num_classes,
                                                          [None] + list(train_videos.shape[1:]),
                                                          [None] + list(train_classes.shape[1:]),
                                                          "logs/lstm" + T, sess)
        videoClassifier.train(train_videos, train_classes, test_videos, test_classes,
                              tf.train.RMSPropOptimizer(learning_rate=args.learning_rate),
                              args.batch_size, args.epochs, sess)

        print("Generating the confusion matrix...")
        predictions = []
        for idx in range(0, test_videos.shape[0], args.batch_size):
            # Extract the following batch_size indices:
            L = min(idx+args.batch_size, test_videos.shape[0])
            predictions.extend(np.argmax(videoClassifier.predict_batch(test_videos[idx:L], sess), axis=1))

        cnf_matrix = sklearn.metrics.confusion_matrix(test_classes, predictions)
        plot_confusion_matrix(cnf_matrix, classes=range(1, len(CLASS_NAMES)+1), title=T)
