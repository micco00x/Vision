import tensorflow as tf
import numpy as np
import os

class VideoClassifier:
    # Define the computational graph with lstm_hidden nodes for the LSTM,
    # num_classes for the softmax and shape for the inputs X (videos), y (classes):
    def __init__(self, lstm_hidden, num_classes, videos_shape, classes_shape):

        # Count number of total epochs in case of multiple trainings:
        self.tot_epochs = 0

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

        # Accuracy of the network:
        correct_pred = tf.equal(tf.argmax(self.prediction, axis=1), tf.argmax(self.y_one_hot, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the graph:
        tf.global_variables_initializer().run()

    # TensorBoard:
    def set_tensorboard_folder(self, tensorboard_folder, sess):
        self.tensorboard_writer = tf.summary.FileWriter(tensorboard_folder, sess.graph)

    # Checkpoint folder:
    def set_checkpoint_folder(self, checkpoint_folder):
        self.checkpoint_folder = checkpoint_folder
        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

    # Train the LSTM on the dataset:
    def train(self,
              train_videos, train_classes, val_videos, val_classes,
              optimizer, batch_size, epochs,
              sess, verbose=True, save_checkpoint=10):

        # Optimize the loss function:
        self.train_step = optimizer.minimize(self.loss)
        sess.run(tf.variables_initializer(optimizer.variables()))

        #for epoch in range(epochs):
        while self.tot_epochs < epochs:
            self.tot_epochs += 1
            print("Epoch {}/{}:".format(self.tot_epochs, epochs))

            # Update tot_loss and tot_accuracy and log to TensorBoard (training set):
            tot_loss, tot_accuracy = self._iterate_dataset("train", train_videos, train_classes, batch_size, sess, verbose)

            if self.tensorboard_writer:
                # TODO: move this inside the graph.
                summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=tot_loss)])
                self.tensorboard_writer.add_summary(summary, self.tot_epochs)
                summary = tf.Summary(value=[tf.Summary.Value(tag="train_accuracy", simple_value=tot_accuracy)])
                self.tensorboard_writer.add_summary(summary, self.tot_epochs)

            # Update tot_loss and tot_accuracy and log to TensorBoard (validation set):
            tot_loss, tot_accuracy = self._iterate_dataset("eval", val_videos, val_classes, batch_size, sess, verbose)

            if self.tensorboard_writer:
                # TODO: move this inside the graph.
                summary = tf.Summary(value=[tf.Summary.Value(tag="val_loss", simple_value=tot_loss)])
                self.tensorboard_writer.add_summary(summary, self.tot_epochs)
                summary = tf.Summary(value=[tf.Summary.Value(tag="val_accuracy", simple_value=tot_accuracy)])
                self.tensorboard_writer.add_summary(summary, self.tot_epochs)

            # Save checkpoint:
            if self.checkpoint_folder and self.tot_epochs % save_checkpoint == 0:
                self.save(os.path.join(self.checkpoint_folder, "epoch_" + str(self.tot_epochs) + ".ckpt"), sess)

    # Predict a video of shape [num frames, ...]
    def predict(self, video, sess):
        return sess.run(self.prediction, feed_dict={self.X: [video]})

    # Predict a batch of videos of shape [batch_size, num frames, ...]
    def predict_batch(self, videos, sess):
        return sess.run(self.prediction, feed_dict={self.X: videos})

    # Save the model specifying a path:
    def save(self, path, sess):
        saver = tf.train.Saver()
        saver.save(sess, path)

    # Load the model specifying a path:
    def load(self, path, sess):
        saver = tf.train.Saver()
        saver.restore(sess, path)

    def _iterate_dataset(self, mode, videos, classes, batch_size, sess, verbose):

        if mode == "train":
            loss_tag = "train_loss"
            accuracy_tag = "train_accuracy"
        else:
            loss_tag = "val_loss"
            accuracy_tag = "val_accuracy"

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

        # SGD random indices:
        new_indices = np.random.randint(videos.shape[0], size=videos.shape[0])

        # Iterate through the dataset:
        for idx in range(0, videos.shape[0], batch_size):

            # Extract the following batch_size indices:
            L = min(idx+batch_size, videos.shape[0])

            if mode == "train":
                rand_idx = new_indices[idx:L]
            else:
                rand_idx = range(idx, L)

            minibatch_X = videos[rand_idx,:]
            minibatch_y = classes[rand_idx]

            # Perform a training step:
            if mode == "train":
                vloss, vaccuracy, bs, _ = sess.run([self.loss, self.accuracy, self.batch_size, self.train_step], feed_dict={self.X: minibatch_X, self.y: minibatch_y})
            else:
                vloss, vaccuracy, bs = sess.run([self.loss, self.accuracy, self.batch_size], feed_dict={self.X: minibatch_X, self.y: minibatch_y})

            # Update tot_loss and tot_accuracy:
            N = N + 1
            tot_loss = (batch_size * sum_loss + bs * vloss) / (N * batch_size + bs)
            sum_loss = sum_loss + vloss
            tot_accuracy = (batch_size * sum_accuracy + bs * vaccuracy) / (N * batch_size + bs)
            sum_accuracy = sum_accuracy + vaccuracy

            if verbose:
                print("Progress: {}/{} - {}: {:2.3} - {}: {:2.3}".format(L, videos.shape[0], loss_tag, tot_loss, accuracy_tag, tot_accuracy), end="\r")
        if verbose:
            print("Progress: {}/{} - {}: {:2.3} - {}: {:2.3}".format(videos.shape[0], videos.shape[0], loss_tag, tot_loss, accuracy_tag, tot_accuracy))

        return tot_loss, tot_accuracy
