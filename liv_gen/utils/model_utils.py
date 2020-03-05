'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
import os
import pickle

from keras.utils import plot_model


class ModelManager():
    '''Class to manage a model.'''

    def __init__(self, data_dir, name):
        self.__folder = os.path.join(data_dir, name)

        if not os.path.exists(self.__folder):
            os.makedirs(os.path.join(self.__folder, 'viz'))
            os.makedirs(os.path.join(self.__folder, 'images'))
            os.makedirs(os.path.join(self.__folder, 'weights'))

    def get_folder(self):
        '''Get folder.'''
        return self.__folder

    def init(self, obj, build):
        '''Initialise model.'''
        if build:
            self.save(obj)
        else:
            self.__load_weights(obj)

    def load(self, model_class):
        '''Load model.'''
        with open(os.path.join(self.__folder, 'params.pkl'), 'rb') as fle:
            params = pickle.load(fle)

        obj = model_class(*params)

        self.__load_weights(obj)

        return obj

    def save(self, obj):
        '''Save model.'''
        with open(os.path.join(self.__folder, 'params.pkl'), 'wb') as fle:
            pickle.dump(obj.get_arguments(), fle)

        self.__plot(obj)

    def __load_weights(self, obj):
        '''Load weights.'''
        obj.get_model().load_weights(
            os.path.join(self.__folder, 'weights/weights.h5'))

    def __plot(self, obj):
        '''Plot model.'''
        for model, name in zip(*[[obj.get_model(),
                                  obj.get_encoder(),
                                  obj.get_decoder()],
                                 ['model', 'encoder', 'decoder']]):
            plot_model(model,
                       to_file=os.path.join(
                           self.__folder, 'viz/%s.png' % name),
                       show_shapes=True,
                       show_layer_names=True)
