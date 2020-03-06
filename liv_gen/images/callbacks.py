'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=arguments-differ
# pylint: disable=dangerous-default-value
# pylint: disable=super-init-not-called
# pylint: disable=wrong-import-order
import os

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

from liv_gen.utils.callbacks import step_decay_schedule
import matplotlib.pyplot as plt
import numpy as np


class ImageCallback(Callback):
    '''Class to implement image-writing Callback.'''

    def __init__(self, obj, folder, print_batch):
        self.__obj = obj
        self.__folder = folder
        self.__print_batch = print_batch
        self.__epoch = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.__epoch = epoch

    def on_batch_end(self, btch, logs={}):
        if btch % self.__print_batch == 0:
            z_new = np.random.normal(size=(1, self.__obj.z_dim))
            reconst = self.__obj.get_decoder().predict(
                np.array(z_new))[0].squeeze()

            filepath = os.path.join(
                self.__folder,
                'images',
                'img_' + str(self.__epoch).zfill(3) + '_' + str(btch) + '.jpg')

            if len(reconst.shape) == 2:
                plt.imsave(filepath, reconst, cmap='gray_r')
            else:
                plt.imsave(filepath, reconst)


def get_callbacks(obj, folder, print_batch, lr_decay):
    '''Get callbacks.'''
    checkpoint1 = ModelCheckpoint(
        os.path.join(folder, 'weights/weights-{epoch:03d}.h5'),
        save_weights_only=True, verbose=1)

    checkpoint2 = ModelCheckpoint(
        os.path.join(folder, 'weights/weights.h5'),
        save_weights_only=True, verbose=1)

    lr_sched = step_decay_schedule(
        initial_lr=obj.learning_rate, decay_factor=lr_decay, step_size=1)

    image_callback = ImageCallback(obj, folder, print_batch)

    return [checkpoint1, checkpoint2, lr_sched, image_callback]
