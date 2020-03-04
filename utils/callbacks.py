'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=arguments-differ
# pylint: disable=dangerous-default-value
# pylint: disable=super-init-not-called
import os

from keras.callbacks import Callback, LearningRateScheduler

import matplotlib.pyplot as plt
import numpy as np


class CustomCallback(Callback):
    '''Class to implement custom Callback.'''

    def __init__(self, run_folder, print_every_n_batches, initial_epoch, ae):
        self.__epoch = initial_epoch
        self.__run_folder = run_folder
        self.__print_every_n_batches = print_every_n_batches
        self.__ae = ae

    def on_batch_end(self, btch, logs={}):
        if btch % self.__print_every_n_batches == 0:
            z_new = np.random.normal(size=(1, self.__ae.z_dim))
            reconst = self.__ae.decoder.predict(np.array(z_new))[0].squeeze()

            filepath = os.path.join(
                self.__run_folder,
                'images',
                'img_' + str(self.__epoch).zfill(3) + '_' + str(btch) + '.jpg')

            if len(reconst.shape) == 2:
                plt.imsave(filepath, reconst, cmap='gray_r')
            else:
                plt.imsave(filepath, reconst)

    def on_epoch_begin(self, epoch, logs={}):
        self.__epoch += 1


def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    '''
    Wrapper func to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)
