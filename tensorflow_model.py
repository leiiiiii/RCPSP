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
        discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ],name="discounted_episode_rewards_")
        # mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")

    with tf.name_scope("hidden"):
        hidden = tf.contrib.layers.fully_connected(inputs=states, num_outputs=10, activation_fn=tf.nn.leaky_relu,name="hidden")

    with tf.name_scope("output"):
        pg_output = tf.contrib.layers.fully_connected(inputs=hidden, num_outputs=output_size, activation_fn=tf.nn.softmax,name="output")

    return pg_output


def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards


class createNeuralNetworkModel:

    # todo list: def discounted_episode_rewards
    #            feed_dict
    #            parameters: gamma, input, output, states, actions(labels)
    #            put the placeholder from function into class

    def __init__(self, input, output, input_size, output_size, learningRate):
        self.input_size = input_size
        self.output_size = output_size
        self.learningRate = learningRate

        # construct only neural network, no feed data
        self.pg_output = build_pg_network(input_size, output_size)
        self.get_parameters = self.return_net_params()

    # ===================================
    # training function part
    # ===================================
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        prob_act = self.pg_output.eval(feed_dict={states: input})

        self._get_act_prob = prob_act

    # --------  Supervised Learning  --------
        with tf.name_scope("su_loss"):
            # loss function is mean square error
            su_loss = tf.losses.softmax_cross_entropy(prob_act, actions)
            self.su_loss = su_loss.mean()

        with tf.name_scope("su_train"):
            self.su_train = tf.train.AdamOptimizer(learningRate).minimize(su_loss)


    # --------  Policy Gradient reinforcement learning  --------
        with tf.name_scope("re_loss"):
            # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
            log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits= prob_act, labels=actions)
            self.re_loss = tf.reduce_mean(log_prob * discounted_episode_rewards_)

        with tf.name_scope("re_train"):
            self.re_train = tf.train.AdamOptimizer(learningRate).minimize(re_loss)


    # --------  Supervised Learning training and test --------
    def su_train(self, states, actions):
        su_loss, prob_act = self.re_train(states, actions)???
        return np.sqrt(su_loss), prob_act

    def su_test(self, states, actions):
        su_loss, prob_act = self.re_train(states, actions)???
        return np.sqrt(su_loss), prob_act

    #  -------- Save/Load network parameters --------
    #this part should be in a with session part.
    def return_net_params(self):
        #create a saver object to save all variables
        #tf.get_collection(tf.Graphkeys.VARIABLES)
        saver = tf.train.Saver


    def set_net_params(self, net_params):
        # First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('save-model.meta')
        w1 = graph.get_tensor_by_name("w1:0")
        w2 = graph.get_tensor_by_name("w2:0")
        op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
        saver.restore(sess, tf.train.latest_checkpoint('./'))

