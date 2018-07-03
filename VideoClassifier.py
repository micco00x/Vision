import tensorflow as tf
import numpy as np

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
              train_videos, train_classes, val_videos, val_classes,
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
                for idx in range(0, train_videos.shape[0], batch_size):
                    # Extract the following args.batch_size indices and perform a training step:
                    L = min(idx+batch_size, train_videos.shape[0])
                    rand_idx = new_indices[idx:L]
                    minibatch_X = train_videos[rand_idx,:,:,:,:]
                    minibatch_y = train_classes[rand_idx]
                    vloss, vaccuracy, bs, _ = sess.run([self.loss, accuracy, self.batch_size, train_step], feed_dict={self.X: minibatch_X, self.y: minibatch_y})

                    # Update tot_loss and tot_accuracy:
                    N = N + 1
                    tot_loss = (batch_size * sum_loss + bs * vloss) / (N * batch_size + bs)
                    sum_loss = sum_loss + vloss
                    tot_accuracy = (batch_size * sum_accuracy + bs * vaccuracy) / (N * batch_size + bs)
                    sum_accuracy = sum_accuracy + vaccuracy

                    print("Progress: {}/{} - {}: {:2.3} - {}: {:2.3}".format(L, train_videos.shape[0], tensorboard_train_loss_tag, tot_loss, tensorboard_train_accuracy_tag, tot_accuracy), end="\r")
                print("Progress: {}/{} - {}: {:2.3} - {}: {:2.3}".format(train_videos.shape[0], train_videos.shape[0], tensorboard_train_loss_tag, tot_loss, tensorboard_train_accuracy_tag, tot_accuracy))

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
                    minibatch_X = val_videos[idx:L,:,:,:,:]
                    minibatch_y = val_classes[idx:L]
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

    #def _iterate_dataset(self):
