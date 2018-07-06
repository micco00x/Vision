import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import sklearn
import numpy as np
import argparse

import VideoClassifier

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
    parser.add_argument("--dataset", help="Dataset (.npz file) that stores tensors used to evaluate the network")
    parser.add_argument("--lstm_hidden", default=256, type=int, help="Size of the hidden layer for the LSTM")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint of the model to evaluate")
    parser.add_argument("--batch_size", default=32, type=int, help="Size of the batch for the LSTM (default=32)")
    parser.add_argument("--cm_title", default="confusionmatrix", help="Title and filename of the confusion matrix")
    args = parser.parse_args()

    # Load tensors (train):
    npzfile = np.load(args.dataset)
    videos = npzfile["videos"]
    classes = npzfile["videos_y"]

    with tf.Session() as sess:
        videoClassifier = VideoClassifier.VideoClassifier(args.lstm_hidden, len(set(classes)),
                                                          [None] + list(videos.shape[1:]),
                                                          [None] + list(classes.shape[1:]),
                                                          None, None, sess)
        videoClassifier.load(args.checkpoint, sess)

        print("Generating the confusion matrix...")
        predictions = []
        for idx in range(0, videos.shape[0], args.batch_size):
            # Extract the following batch_size indices:
            L = min(idx+args.batch_size, videos.shape[0])
            predictions.extend(np.argmax(videoClassifier.predict_batch(videos[idx:L], sess), axis=1))

        cnf_matrix = sklearn.metrics.confusion_matrix(classes, predictions)
        plot_confusion_matrix(cnf_matrix, classes=range(1, len(CLASS_NAMES)+1), title=args.cm_title)
