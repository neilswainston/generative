'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=no-self-use
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=invalid-name
import os
import pickle

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, \
    Reshape, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

import numpy as np
from utils.callbacks import CustomCallback, step_decay_schedule


class Autoencoder():
    '''Class to represent an Autoencoder.'''

    def __init__(self,
                 name,
                 input_dim,
                 encoder_conv_filters,
                 encoder_conv_kernel_size,
                 encoder_conv_strides,
                 decoder_conv_t_filters,
                 decoder_conv_t_kernel_size,
                 decoder_conv_t_strides,
                 z_dim,
                 learning_rate,
                 use_batch_norm=False,
                 use_dropout=False):

        self.name = name
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)

        self.__build()

    def __build(self):

        # THE ENCODER
        encoder_input = Input(shape=self.input_dim, name='encoder_input')

        x = encoder_input

        for i in range(self.n_layers_encoder):
            conv_layer = Conv2D(
                filters=self.encoder_conv_filters[i],
                kernel_size=self.encoder_conv_kernel_size[i],
                strides=self.encoder_conv_strides[i],
                padding='same',
                name='encoder_conv_' + str(i)
            )

            x = conv_layer(x)

            x = LeakyReLU()(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]
        x = Flatten()(x)
        encoder_output = self.get_enc_output(x)

        self.encoder = Model(encoder_input, encoder_output)

        # THE DECODER
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv2DTranspose(
                filters=self.decoder_conv_t_filters[i],
                kernel_size=self.decoder_conv_t_kernel_size[i],
                strides=self.decoder_conv_t_strides[i],
                padding='same',
                name='decoder_conv_t_' + str(i)
            )

            x = conv_t_layer(x)

            if i < self.n_layers_decoder - 1:
                x = LeakyReLU()(x)

                if self.use_batch_norm:
                    x = BatchNormalization()(x)

                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation('sigmoid')(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        # THE FULL AUTOENCODER
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)

    def compile(self):
        '''Compile.'''
        self.model.compile(optimizer=Adam(lr=self.learning_rate),
                           loss=self.r_loss)

    def train(self, x_train, batch_size, epochs, run_folder,
              print_every_n_batches=100, initial_epoch=0, lr_decay=1):
        '''Train.'''

        self.model.fit(
            x_train,
            x_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=self.__get_callbacks(run_folder,
                                           print_every_n_batches,
                                           initial_epoch,
                                           lr_decay),
        )

    def train_with_generator(self, data_flow, epochs, steps_per_epoch,
                             run_folder,
                             print_every_n_batches=100,
                             initial_epoch=0, lr_decay=1):
        '''Train with generator.'''
        self.model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))

        self.model.fit_generator(
            data_flow,
            shuffle=True,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=self.__get_callbacks(run_folder,
                                           print_every_n_batches,
                                           initial_epoch,
                                           lr_decay),
            steps_per_epoch=steps_per_epoch
        )

    def plot_model(self, run_folder):
        '''Plot model.'''
        plot_model(self.model,
                   to_file=os.path.join(run_folder, 'viz/model.png'),
                   show_shapes=True,
                   show_layer_names=True)
        plot_model(self.encoder,
                   to_file=os.path.join(run_folder, 'viz/encoder.png'),
                   show_shapes=True,
                   show_layer_names=True)
        plot_model(self.decoder,
                   to_file=os.path.join(run_folder, 'viz/decoder.png'),
                   show_shapes=True,
                   show_layer_names=True)

    def save(self, folder):
        '''Save.'''
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as fle:
            pickle.dump(self.__get_arguments(), fle)

        self.plot_model(folder)

    def load_weights(self, filepath):
        '''Load weights.'''
        self.model.load_weights(filepath)

    def r_loss(self, y_true, y_pred):
        '''Get loss.'''
        return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

    def get_enc_output(self, x):
        '''Get encoder output.'''
        return Dense(self.z_dim, name='encoder_output')(x)

    def __get_callbacks(self, run_folder, print_every_n_batches, initial_epoch,
                        lr_decay):
        '''Get callbacks.'''
        custom_callback = CustomCallback(
            run_folder, print_every_n_batches, initial_epoch, self)

        lr_sched = step_decay_schedule(
            initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint_filepath = os.path.join(
            run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")

        checkpoint1 = ModelCheckpoint(
            checkpoint_filepath, save_weights_only=True, verbose=1)

        checkpoint2 = ModelCheckpoint(
            os.path.join(run_folder, 'weights/weights.h5'),
            save_weights_only=True, verbose=1)

        return [checkpoint1, checkpoint2, custom_callback, lr_sched]

    def __get_arguments(self):
        '''Get arguments.'''
        return [
            self.name,
            self.input_dim,
            self.encoder_conv_filters,
            self.encoder_conv_kernel_size,
            self.encoder_conv_strides,
            self.decoder_conv_t_filters,
            self.decoder_conv_t_kernel_size,
            self.decoder_conv_t_strides,
            self.z_dim,
            self.learning_rate,
            self.use_batch_norm,
            self.use_dropout
        ]
