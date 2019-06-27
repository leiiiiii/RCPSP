#Neural Network Model using package keras
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam


def createNeuralNetworkModel(input_size, output_size):

    model = Sequential()

    model.add(Dense(units=128, activation='relu', input_dim=input_size))
    model.add(Dropout(0.8))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(output_size,activation='softmax'))
    adam = Adam(lr=0.001, decay = 1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    #fit function same as tensorflow
    return model

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))

        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')#plt.plot(x,y),plot acc as graph
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')

        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')

        plt.legend(loc="upper right")
        plt.show()

history = LossHistory()









