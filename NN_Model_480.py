import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.optimizers import SGD
from tflearn.layers.estimator import regression
from datetime import datetime
import numpy as np

timestamp = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())



def createNeuralNetworkModel(input_size, output_size, learningRate):
    network = input_data(shape=[None, input_size], name="input")

    #tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

    network = fully_connected(network, 10, activation="relu")

    network = fully_connected(network, 10, activation="relu")

    network = fully_connected(network, 10, activation="relu")

    network = fully_connected(network, output_size, activation="softmax")

    network = regression(network, optimizer='sgd', learning_rate=learningRate, loss="categorical_crossentropy",name="targets")

    #model = tflearn.DNN(network,tensorboard_dir='log/' + timestamp, tensorboard_verbose=1)
    model = tflearn.DNN(network, tensorboard_verbose=0)

    return model






