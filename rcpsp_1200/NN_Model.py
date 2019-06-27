import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from datetime import datetime

timestamp = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())

def createNeuralNetworkModel(input_size, output_size, learningRate):
    network = input_data(shape=[None, input_size], name="input")

    network = fully_connected(network, 10, activation="relu")
    network = dropout(network, 0.8)

    # network = fully_connected(network, 128, activation="relu")
    # network = dropout(network, 0.8)
    #
    # network = fully_connected(network, 256, activation="relu")
    # network = dropout(network, 0.8)
    #
    # network = fully_connected(network, 128, activation="relu")
    # network = dropout(network, 0.8)
    #
    # network = fully_connected(network, 64, activation="relu")
    # network = dropout(network, 0.8)

    network = fully_connected(network, output_size, activation="softmax")

    network = regression(network, optimizer="adam", learning_rate=learningRate, loss="categorical_crossentropy", name="targets")

    model = tflearn.DNN(network,tensorboard_dir='log/' + timestamp, tensorboard_verbose=3)

    return model