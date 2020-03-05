'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
from keras.callbacks import LearningRateScheduler
import numpy as np


def step_decay_schedule(initial_lr, decay_factor, step_size):
    '''
    Wrapper func to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)
