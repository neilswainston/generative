'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=unused-argument
from keras import backend as K
from keras.layers import Dense, Lambda
from keras.optimizers import Adam
from models.ae import Autoencoder


class VariationalAutoencoder(Autoencoder):
    '''Class to represent a Variational Autoencoder.'''

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
                 r_loss_factor,
                 use_batch_norm=False,
                 use_dropout=False):

        self.__r_loss_factor = r_loss_factor
        self.__mu = None
        self.__log_var = None

        super().__init__(
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
            use_batch_norm,
            use_dropout)

    def compile(self):
        '''Compile.'''
        self.model.compile(optimizer=Adam(lr=self.learning_rate),
                           loss=self.__vae_loss,
                           metrics=[self.r_loss, self.__vae_kl_loss])

    def r_loss(self, y_true, y_pred):
        '''Get loss.'''
        return self.__r_loss_factor * super().r_loss(y_true, y_pred)

    def get_enc_output(self, x):
        '''Get encoder output.'''
        self.__mu = Dense(self.z_dim, name='mu')(x)
        self.__log_var = Dense(self.z_dim, name='log_var')(x)

        return Lambda(_sampling, name='encoder_output')(
            [self.__mu, self.__log_var])

    def __vae_kl_loss(self, y_true, y_pred):
        return -0.5 * K.sum(1 + self.__log_var - K.square(self.__mu) -
                            K.exp(self.__log_var), axis=1)

    def __vae_loss(self, y_true, y_pred):
        r_loss = self.r_loss(y_true, y_pred)
        kl_loss = self.__vae_kl_loss(y_true, y_pred)
        return r_loss + kl_loss

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
            self.__r_loss_factor,
            self.use_batch_norm,
            self.use_dropout
        ]


def _sampling(args):
    mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
    return mu + K.exp(log_var / 2) * epsilon
