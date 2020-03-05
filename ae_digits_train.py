'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
import os

from models.ae import Autoencoder
from utils.loaders import load_mnist


# ## Set parameters
# run params
SECTION = 'ae'
RUN_ID = '0001'
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

    # ## Load the data
    (x_train, _), _ = load_mnist()

    # ## Define the structure of the neural network
    model = Autoencoder(
        name='autoencoder',
        input_dim=(28, 28, 1),
        encoder_conv_filters=[32, 64, 64, 64],
        encoder_conv_kernel_size=[3, 3, 3, 3],
        encoder_conv_strides=[1, 2, 2, 1],
        decoder_conv_t_filters=[64, 64, 32, 1],
        decoder_conv_t_kernel_size=[3, 3, 3, 3],
        decoder_conv_t_strides=[1, 2, 2, 1],
        z_dim=2,
        learning_rate=0.0005
    )

    if mode == 'build':
        model.save(RUN_FOLDER)
    else:
        model.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

    model.encoder.summary()
    model.decoder.summary()

    # ## Train the autoencoder:
    model.compile()

    model.train(
        x_train[:1000],
        batch_size=32,
        epochs=200,
        run_folder=RUN_FOLDER,
        initial_epoch=0
    )


if __name__ == '__main__':
    main()
