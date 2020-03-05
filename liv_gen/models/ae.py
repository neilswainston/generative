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

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, \
    Reshape, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras.optimizers import Adam

from liv_gen.utils.callbacks import ImageCallback, step_decay_schedule
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
                           loss=self.r_loss)

    def train(self, x_train, batch_size, epochs, folder,
              print_batch=100, lr_decay=1):
        '''Train.'''
        self.model.fit(
            x_train,
            x_train,
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=self.__get_callbacks(folder, print_batch, lr_decay),
        )

    def train_with_generator(self, data_flow, epochs, steps_per_epoch,
                             folder,
                             print_batch=100,
                             lr_decay=1):
        '''Train with generator.'''
        self.model.fit_generator(
            data_flow,
            shuffle=True,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=self.__get_callbacks(folder, print_batch, lr_decay),
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

    def r_loss(self, y_true, y_pred):
        '''Get loss.'''
        return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

    def get_enc_output(self, x):
        '''Get encoder output.'''
        return Dense(self.z_dim, name='encoder_output')(x)

    def __get_callbacks(self, folder, print_batch, lr_decay):
        '''Get callbacks.'''
        image_callback = ImageCallback(folder, print_batch, self)

        lr_sched = step_decay_schedule(
            initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint1 = ModelCheckpoint(
            os.path.join(
                folder, 'weights/weights-{epoch:03d}-{loss:.2f}.h5'),
            save_weights_only=True, verbose=1)

        checkpoint2 = ModelCheckpoint(
            os.path.join(folder, 'weights/weights.h5'),
            save_weights_only=True, verbose=1)

        return [checkpoint1, checkpoint2, image_callback, lr_sched]
