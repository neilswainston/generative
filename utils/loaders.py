'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
import os
import pickle
from keras.datasets import mnist


def load_model(model_class, folder):
    '''Local model.'''
    with open(os.path.join(folder, 'params.pkl'), 'rb') as fle:
        params = pickle.load(fle)

    model = model_class(*params)

    model.load_weights(os.path.join(folder, 'weights/weights.h5'))

    return model


def load_mnist():
    '''Load mnist.'''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))

    return (x_train, y_train), (x_test, y_test)
