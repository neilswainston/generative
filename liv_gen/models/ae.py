'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=no-self-use
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=wrong-import-order
from keras import backend as K
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, \
    Reshape, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras.optimizers import Adam
import numpy as np


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

        self.__callbacks = []

        self.__build()

    def __build(self):
        # THE ENCODER
        encoder_input = Input(shape=self.input_dim, name='encoder_input')

        x = encoder_input

        for i in range(len(self.encoder_conv_filters)):
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

        for i in range(len(self.decoder_conv_t_filters)):
            conv_t_layer = Conv2DTranspose(
                filters=self.decoder_conv_t_filters[i],
                kernel_size=self.decoder_conv_t_kernel_size[i],
                strides=self.decoder_conv_t_strides[i],
                padding='same',
                name='decoder_conv_t_' + str(i)
            )

            x = conv_t_layer(x)

            if i < len(self.decoder_conv_t_filters) - 1:
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
                           loss=r_loss)

    def add_callbacks(self, callbacks):
        '''Add callbacks.'''
        self.__callbacks.extend(callbacks)

    def train(self, x_train, batch_size, epochs):
        '''Train.'''
        self.model.fit(
            x_train,
            x_train,
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.__callbacks,
        )

    def train_with_generator(self, data_flow, epochs, steps_per_epoch):
        '''Train with generator.'''
        self.model.fit_generator(
            data_flow,
            shuffle=True,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=self.__callbacks,
        )

    def get_arguments(self):
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

    def get_model(self):
        '''Get model.'''
        return self.model

    def get_encoder(self):
        '''Get encoder.'''
        return self.encoder

    def get_decoder(self):
        '''Get decoder.'''
        return self.decoder

    def get_enc_output(self, x):
        '''Get encoder output.'''
        return Dense(self.z_dim, name='encoder_output')(x)


def r_loss(y_true, y_pred):
    '''Get loss.'''
    return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
