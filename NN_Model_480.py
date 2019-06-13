import tflearn
#import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.optimizers import SGD
from tflearn.layers.estimator import regression
from datetime import datetime

timestamp = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())

def createNeuralNetworkModel(input_size, output_size, learningRate):
    network = input_data(shape=[None, input_size], name="input")

    # network = fully_connected(network, 10, activation="leakyrelu", weights_init=tf.constant_initializer(value=0.01))
    network = fully_connected(network, 10, activation="leakyrelu")
    # network = dropout(network, 0.8)

    # network = fully_connected(network, output_size, activation="softmax", weights_init=tf.constant_initializer(value=0.01))
    network = fully_connected(network, output_size, activation="softmax")
    # sgd = SGD( lr_decay=0.96, decay_step=100)
    # network = regression(network, optimizer="sgd", learning_rate=learningRate, loss="categorical_crossentropy",
    #                      name="targets")
    network = regression(network, optimizer="adam", learning_rate=learningRate, loss="categorical_crossentropy", name="targets")

    model = tflearn.DNN(network,tensorboard_dir='log/' + timestamp, tensorboard_verbose=3)

    return model