#Neural Network Model using package tensorflow
import tensorflow as tf
import numpy as np


# ===================================
# build neural network
# ===================================
def build_pg_network(input_size, output_size):
    with tf.name_scope("input"):
        # Add this placeholder for having this variable in tensorboard
        states = tf.placeholder(tf.float32, [None, input_size], name="states")
        actions = tf.placeholder(tf.float32, [None, output_size], name="actions")
        # discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ],name="discounted_episode_rewards_")
        # mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")

    with tf.name_scope("hidden"):
        hidden = tf.contrib.layers.fully_connected(inputs=states, num_outputs=10, activation_fn=tf.nn.leaky_relu,name="hidden")

    with tf.name_scope("output"):
        output = tf.contrib.layers.fully_connected(inputs=hidden, num_outputs=output_size, activation_fn=tf.nn.softmax,name="output")

    return output


class createNeuralNetworkMode:

    def __init__(self, input_size, output_size, learningRate):
        self.input_size = input_size
        self.output_size = output_size
        self.learningRate = learningRate

        self.output = build_pg_network(input_size, output_size)

        with tf.Session() as sess:
            action_probability_distribution = sess.run(self.output, feed_dict={states: states.reshape([])})

        action = np.random.choice(range(action_probability_distribution.shape[1]),
                                  p=action_probability_distribution.ravel())

    # ===================================
    # training function part
    # ===================================

    # --------  Supervised Learning  --------
        with tf.name_scope("loss"):
            su_loss = lasagne.objectives.categorical_crossentropy(action_probability_distribution , actions)
            su_loss = su_loss.mean()


    with tf.name_scope("train"):

        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)


    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)


# tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                # If you have single-class labels, where an object can only belong to one class, you might now consider using
                # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.

