import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from datetime import datetime
import numpy as np

timestamp = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())



def createNeuralNetworkModel(input_size, output_size, learningRate):
    network = input_data(shape=[None, input_size], name="input")

    #tflearn.init_graph(gpu_memory_fraction=0.2)

    network = fully_connected(network, 32, activation="tanh", regularizer='L2',weights_init=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32),bias_init=tf.constant_initializer(value=0.5))
    network = dropout(network, 0.8)

    network = fully_connected(network, 32, activation="tanh", regularizer='L2',weights_init=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32),bias_init=tf.constant_initializer(value=0.3))
    network = dropout(network, 0.8)

    # network = fully_connected(network, 32, activation="tanh", regularizer='L2',weights_init=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32),bias_init=tf.constant_initializer(value=0.3))
    # network = dropout(network, 0.8)

    # network = fully_connected(network, 128, activation="tanh", regularizer='L2',weights_init=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32),bias_init=tf.constant_initializer(value=0.3))
    # network = dropout(network, 0.8)
    # #
    # network = fully_connected(network, 32, activation="tanh", regularizer='L2',weights_init=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32),bias_init=tf.constant_initializer(value=0.3))
    # network = dropout(network, 0.8)

    network = fully_connected(network, output_size, activation="softmax",regularizer='L2',weights_init=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32),bias_init=tf.constant_initializer(value=0.1))

    network = regression(network, optimizer='RMSProp', learning_rate=learningRate, loss="categorical_crossentropy",name="targets")

    #model = tflearn.DNN(network,tensorboard_dir='log/' + timestamp, tensorboard_verbose=3)
    model = tflearn.DNN(network, tensorboard_verbose=0)

    return model






