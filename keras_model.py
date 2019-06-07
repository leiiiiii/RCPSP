#Neural Network Model using package keras
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam


def createNeuralNetworkModel(input_size, output_size):

    model = Sequential()

    model.add(Dense(units=128, activation='relu', input_dim=input_size))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))

    model.add(Dense(output_size))
    adam = Adam(lr=0.001, decay = 1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    #fit function same as tensorflow
    return model








