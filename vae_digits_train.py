'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
import os

from models.vae import VariationalAutoencoder
from utils.loaders import load_mnist

# run params
SECTION = 'vae'
RUN_ID = '0002'
DATA_NAME = 'digits'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
    os.makedirs(os.path.join(RUN_FOLDER, 'viz'))
    os.makedirs(os.path.join(RUN_FOLDER, 'images'))
    os.makedirs(os.path.join(RUN_FOLDER, 'weights'))


def main():
    '''main method.'''
    mode = 'build'

    # ## data
    (x_train, _), _ = load_mnist()

    # ## architecture
    model = VariationalAutoencoder(
        name='variational_autoencoder',
        input_dim=(28, 28, 1),
        encoder_conv_filters=[32, 64, 64, 64],
        encoder_conv_kernel_size=[3, 3, 3, 3],
        encoder_conv_strides=[1, 2, 2, 1],
        decoder_conv_t_filters=[64, 64, 32, 1],
        decoder_conv_t_kernel_size=[3, 3, 3, 3],
        decoder_conv_t_strides=[1, 2, 2, 1],
        z_dim=2,
        learning_rate=0.0005,
        r_loss_factor=1000
    )

    if mode == 'build':
        model.save(RUN_FOLDER)
    else:
        model.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

    model.encoder.summary()
    model.decoder.summary()

    # ## training
    model.compile()

    model.train(
        x_train,
        batch_size=32,
        epochs=10,
        run_folder=RUN_FOLDER,
        print_every_n_batches=100,
        initial_epoch=0
    )


if __name__ == '__main__':
    main()
