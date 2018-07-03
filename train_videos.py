import numpy as np
import cv2
import argparse
import sys
import os
import tensorflow as tf
import time

DEFAULT_LOGS_DIR = "logs"

class VideoClassifier:
    # Define the computational graph with lstm_hidden nodes for the LSTM,
    # num_classes for the softmax and shape for the inputs X (videos), y (classes):
    def __init__(self, lstm_hidden, num_classes, videos_shape, classes_shape):
        # Input (videos' frames) and output (indices) of the network:
        self.X = tf.placeholder(tf.float32, shape=videos_shape)
        self.y = tf.placeholder(tf.uint8, shape=classes_shape)
        self.batch_size = tf.shape(self.X)[0]

        # Reshape and unstack the input so that it can be fed to the LSTM:
        # Reshape X to [batch_size, num frames, width*height*channels] and then
        # unstack it to a list of length num frames containing tensors of
        # shape [batch_size, width*height*channels]:
        X_reshaped  = tf.reshape(self.X, [self.batch_size, videos_shape[1], np.prod(videos_shape[2:])])
        X_unstacked = tf.unstack(X_reshaped, videos_shape[1], axis=1)

        # LSTM part:
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(lstm_hidden)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=0.2)
        rnn_outputs, rnn_states = tf.contrib.rnn.static_rnn(rnn_cell, X_unstacked, dtype=tf.float32)

        # Output of the correct size:
        out_weights = tf.Variable(tf.random_normal([lstm_hidden, num_classes]))
        out_bias    = tf.Variable(tf.random_normal([num_classes]))
        self.prediction  = tf.matmul(rnn_outputs[-1], out_weights) + out_bias

        # Loss:
        self.y_one_hot = tf.one_hot(self.y, num_classes)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction, labels=self.y_one_hot))

    def train(self,
              videos, classes, val_videos, val_classes,
              learning_rate, batch_size, epochs,
              tensorboard_folder,
              tensorboard_train_loss_tag="train_loss", tensorboard_train_accuracy_tag="train_accuracy",
              tensorboard_test_loss_tag="test_loss", tensorboard_test_accuracy_tag="test_accuracy"):

        # Train the LSTM on the dataset:
        with tf.Session() as sess:

            # Loss and optimizer:
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            train_step = optimizer.minimize(self.loss)

            # Accuracy of the network:
            correct_pred = tf.equal(tf.argmax(self.prediction, axis=1), tf.argmax(self.y_one_hot, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # TensorBoard:
            tensorboard_writer = tf.summary.FileWriter(tensorboard_folder, sess.graph)

            # Initialize the graph:
            tf.global_variables_initializer().run()

            for epoch in range(epochs):
                print("Epoch {}/{}:".format(epoch+1, epochs))
                new_indices = np.random.randint(videos.shape[0], size=videos.shape[0])

                # Let mu_i = vloss for each iteration, let's say the iteration is N,
                # bs is the batch_size of the current iteration,
                # sum_loss = mu_1 + mu_2 + ... + mu_N,
                # tot_loss = (args.batch_size * sum_loss + bs * mu_{N+1}) / (N * args.batch_size + bs).
                # The same thing applies for sum_accuracy and tot_accuracy.
                # NOTE: distinguish between args.batch_size and bs is important in order to
                # consider the right number of elements for each iteration:
                N = 0
                sum_loss = 0
                tot_loss = 0
                sum_accuracy = 0
                tot_accuracy = 0

                # Iterate through the train dataset using new_indices (random shuffle):
                for idx in range(0, videos.shape[0], batch_size):
                    # Extract the following args.batch_size indices and perform a training step:
                    L = min(idx+batch_size, videos.shape[0])
                    rand_idx = new_indices[idx:L]
                    minibatch_X = train_videos[rand_idx,:,:,:,:]
                    minibatch_y = train_videos_y[rand_idx]
                    vloss, vaccuracy, bs, _ = sess.run([self.loss, accuracy, self.batch_size, train_step], feed_dict={self.X: minibatch_X, self.y: minibatch_y})

                    # Update tot_loss and tot_accuracy:
                    N = N + 1
                    tot_loss = (batch_size * sum_loss + bs * vloss) / (N * batch_size + bs)
                    sum_loss = sum_loss + vloss
                    tot_accuracy = (batch_size * sum_accuracy + bs * vaccuracy) / (N * batch_size + bs)
                    sum_accuracy = sum_accuracy + vaccuracy

                    print("Progress: {}/{} - {}: {:2.3} - {}: {:2.3}".format(L, videos.shape[0], tensorboard_train_loss_tag, tot_loss, tensorboard_train_accuracy_tag, tot_accuracy), end="\r")
                print("Progress: {}/{} - {}: {:2.3} - {}: {:2.3}".format(videos.shape[0], videos.shape[0], tensorboard_train_loss_tag, tot_loss, tensorboard_train_accuracy_tag, tot_accuracy))

                # Update tot_loss and tot_accuracy and log to TensorBoard:
                # TODO: move this inside the graph.
                summary = tf.Summary(value=[tf.Summary.Value(tag=tensorboard_train_loss_tag, simple_value=tot_loss)])
                tensorboard_writer.add_summary(summary, epoch)
                summary = tf.Summary(value=[tf.Summary.Value(tag=tensorboard_train_accuracy_tag, simple_value=tot_accuracy)])
                tensorboard_writer.add_summary(summary, epoch)

                # TODO: this second part is identical to the first one except for train_step.
                N = 0
                sum_loss = 0
                tot_loss = 0
                sum_accuracy = 0
                tot_accuracy = 0

                # Iterate through the val dataset using new_indices (random shuffle):
                for idx in range(0, val_videos.shape[0], batch_size):
                    # Extract the following args.batch_size indices and perform a training step:
                    L = min(idx+batch_size, val_videos.shape[0])
                    minibatch_X = test_videos[idx:L,:,:,:,:]
                    minibatch_y = test_videos_y[idx:L]
                    vloss, vaccuracy, bs = sess.run([self.loss, accuracy, self.batch_size], feed_dict={self.X: minibatch_X, self.y: minibatch_y})

                    # Update tot_loss and tot_accuracy:
                    N = N + 1
                    tot_loss = (batch_size * sum_loss + bs * vloss) / (N * batch_size + bs)
                    sum_loss = sum_loss + vloss
                    tot_accuracy = (batch_size * sum_accuracy + bs * vaccuracy) / (N * batch_size + bs)
                    sum_accuracy = sum_accuracy + vaccuracy

                    print("Progress: {}/{} - {}: {:2.3} - test_accuracy: {:2.3}".format(L, val_videos.shape[0], tensorboard_test_loss_tag, tot_loss, tot_accuracy), end="\r")
                print("Progress: {}/{} - {}: {:2.3} - {}: {:2.3}".format(val_videos.shape[0], val_videos.shape[0], tensorboard_test_loss_tag, tot_loss, tensorboard_test_accuracy_tag, tot_accuracy))

                # Update tot_loss and tot_accuracy and log to TensorBoard:
                # TODO: move this inside the graph.
                summary = tf.Summary(value=[tf.Summary.Value(tag=tensorboard_test_loss_tag, simple_value=tot_loss)])
                tensorboard_writer.add_summary(summary, epoch)
                summary = tf.Summary(value=[tf.Summary.Value(tag=tensorboard_test_accuracy_tag, simple_value=tot_accuracy)])
                tensorboard_writer.add_summary(summary, epoch)

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
train_videos_y = npzfile["videos_y"]

# Load tensors (test):
npzfile = np.load(args.test)
test_videos = npzfile["videos"]
test_videos_y = npzfile["videos_y"]

# videos should be [num videos, num frames, width, height, channels],
# videos_y should be [num_videos, ]:
print(train_videos.shape)
print(train_videos_y.shape)
print(test_videos.shape)
print(test_videos_y.shape)

# Convert videos' frames from uint8 to float32, specifically from [0,255] to [0,1]:
train_videos = train_videos.astype(np.float32) / 255
test_videos = test_videos.astype(np.float32) / 255
#print("Checking videos are in range [0,1]:", (0 <= videos).all() and (videos <= 1).all())

# LSTM hyperparameters:
lstm_hidden = args.lstm_hidden
num_classes = len(set(train_videos_y))
learning_rate = args.learning_rate
epochs = args.epochs

videoClassifier = VideoClassifier(lstm_hidden, num_classes, [None] + list(train_videos.shape[1:]), [None] + list(train_videos_y.shape[1:]))
videoClassifier.train(train_videos, train_videos_y, test_videos, test_videos_y,
                      args.learning_rate, args.batch_size, args.epochs,
                      "logs/lstm" + time.strftime("%Y%m%dT%H%M"))
