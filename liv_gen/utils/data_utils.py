'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=too-few-public-methods
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator


def load_mnist():
    '''Load mnist.'''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))

    return (x_train, y_train), (x_test, y_test)


class ImageLabelLoader():
    '''Class to load image labels.'''

    def __init__(self, image_folder, target_size):
        self.image_folder = image_folder
        self.target_size = target_size

    def build(self, att, batch_size, columns=None):
        '''Build.'''
        data_gen = ImageDataGenerator(rescale=1. / 255)

        if columns is not None:
            data_flow = data_gen.flow_from_dataframe(
                att,
                self.image_folder,
                x_col='image_id',
                y_col=columns,
                target_size=self.target_size,
                class_mode='multi_output',
                batch_size=batch_size,
                shuffle=True
            )
        else:
            data_flow = data_gen.flow_from_dataframe(
                att,
                self.image_folder,
                x_col='image_id',
                target_size=self.target_size,
                class_mode='input',
                batch_size=batch_size,
                shuffle=True
            )

        return data_flow
