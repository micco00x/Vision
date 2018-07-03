import numpy as np
import cv2
import argparse
import sys
import os
import tensorflow as tf
import time

DEFAULT_LOGS_DIR = "logs"

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

# Train the LSTM on the dataset:
with tf.Session() as sess:
    # Input (videos' frames) and output (indices) of the network:
    X = tf.placeholder(tf.float32, shape=[None] + list(train_videos.shape[1:]))
    y = tf.placeholder(tf.uint8, shape=[None] + list(train_videos_y.shape[1:]))
    batch_size = tf.shape(X)[0]

    # Reshape and unstack the input so that it can be fed to the LSTM:
    # Reshape X to [batch_size, num frames, width*height*channels] and then
    # unstack it to a list of length num frames containing tensors of
    # shape [batch_size, width*height*channels]:
    X_reshaped  = tf.reshape(X, [batch_size, train_videos.shape[1], np.prod(train_videos.shape[2:])])
    X_unstacked = tf.unstack(X_reshaped, train_videos.shape[1], axis=1)

    # LSTM part:
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(lstm_hidden)
    rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=0.2)
    rnn_outputs, rnn_states = tf.contrib.rnn.static_rnn(rnn_cell, X_unstacked, dtype=tf.float32)

    # Output of the correct size:
    out_weights = tf.Variable(tf.random_normal([lstm_hidden, num_classes]))
    out_bias    = tf.Variable(tf.random_normal([num_classes]))
    prediction  = tf.matmul(rnn_outputs[-1], out_weights) + out_bias

    # Loss and optimizer:
    y_one_hot = tf.one_hot(y, num_classes)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y_one_hot))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss)

    # Accuracy of the network:
    correct_pred = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y_one_hot, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # TensorBoard:
    train_writer = tf.summary.FileWriter("logs/lstm" + time.strftime("%Y%m%dT%H%M"), sess.graph)

    # Initialize the graph:
    tf.global_variables_initializer().run()

    # Train the network:
    for epoch in range(epochs):
        print("Epoch {}/{}:".format(epoch+1, epochs))
        new_indices = np.random.randint(train_videos.shape[0], size=train_videos.shape[0])

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
        for idx in range(0, train_videos.shape[0], args.batch_size):
            # Extract the following args.batch_size indices and perform a training step:
            L = min(idx+args.batch_size, train_videos.shape[0])
            rand_idx = new_indices[idx:L]
            minibatch_X = train_videos[rand_idx,:,:,:,:]
            minibatch_y = train_videos_y[rand_idx]
            vloss, vaccuracy, bs, _ = sess.run([loss, accuracy, batch_size, train_step], feed_dict={X: minibatch_X, y: minibatch_y})

            # Update tot_loss and tot_accuracy:
            N = N + 1
            tot_loss = (args.batch_size * sum_loss + bs * vloss) / (N * args.batch_size + bs)
            sum_loss = sum_loss + vloss
            tot_accuracy = (args.batch_size * sum_accuracy + bs * vaccuracy) / (N * args.batch_size + bs)
            sum_accuracy = sum_accuracy + vaccuracy

            print("Progress: {}/{} - train_loss: {:2.3} - train_accuracy: {:2.3}".format(L, train_videos.shape[0], tot_loss, tot_accuracy), end="\r")
        print("Progress: {}/{} - train_loss: {:2.3} - train_accuracy: {:2.3}".format(train_videos.shape[0], train_videos.shape[0], tot_loss, tot_accuracy))

        # Update tot_loss and tot_accuracy and log to TensorBoard:
        # TODO: move this inside the graph.
        summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=tot_loss)])
        train_writer.add_summary(summary, epoch)
        summary = tf.Summary(value=[tf.Summary.Value(tag="train_accuracy", simple_value=tot_accuracy)])
        train_writer.add_summary(summary, epoch)

        # TODO: this second part is identical to the first one except for train_step.
        N = 0
        sum_loss = 0
        tot_loss = 0
        sum_accuracy = 0
        tot_accuracy = 0

        # Iterate through the test dataset using new_indices (random shuffle):
        for idx in range(0, test_videos.shape[0], args.batch_size):
            # Extract the following args.batch_size indices and perform a training step:
            L = min(idx+args.batch_size, test_videos.shape[0])
            minibatch_X = test_videos[idx:L,:,:,:,:]
            minibatch_y = test_videos_y[idx:L]
            vloss, vaccuracy, bs = sess.run([loss, accuracy, batch_size], feed_dict={X: minibatch_X, y: minibatch_y})

            # Update tot_loss and tot_accuracy:
            N = N + 1
            tot_loss = (args.batch_size * sum_loss + bs * vloss) / (N * args.batch_size + bs)
            sum_loss = sum_loss + vloss
            tot_accuracy = (args.batch_size * sum_accuracy + bs * vaccuracy) / (N * args.batch_size + bs)
            sum_accuracy = sum_accuracy + vaccuracy

            print("Progress: {}/{} - test_loss: {:2.3} - test_accuracy: {:2.3}".format(L, test_videos.shape[0], tot_loss, tot_accuracy), end="\r")
        print("Progress: {}/{} - test_loss: {:2.3} - test_accuracy: {:2.3}".format(test_videos.shape[0], test_videos.shape[0], tot_loss, tot_accuracy))

        # Update tot_loss and tot_accuracy and log to TensorBoard:
        # TODO: move this inside the graph.
        summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=tot_loss)])
        train_writer.add_summary(summary, epoch)
        summary = tf.Summary(value=[tf.Summary.Value(tag="test_accuracy", simple_value=tot_accuracy)])
        train_writer.add_summary(summary, epoch)
