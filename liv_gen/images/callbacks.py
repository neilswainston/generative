'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=arguments-differ
# pylint: disable=dangerous-default-value
# pylint: disable=super-init-not-called
import os

from keras.callbacks import Callback

import matplotlib.pyplot as plt
import numpy as np


class ImageCallback(Callback):
    '''Class to implement image-writing Callback.'''

    def __init__(self, folder, print_batch, obj):
        self.__epoch = 0
        self.__folder = folder
        self.__print_batch = print_batch
        self.__obj = obj

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
