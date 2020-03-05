'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
import os.path
import sys

from liv_gen.images import digits
from liv_gen.models.vae import VariationalAutoencoder


def main(args):
    '''main method.'''
    obj = VariationalAutoencoder(
        name=os.path.splitext(os.path.basename(__file__))[0],
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

    digits.run(obj, args[0] == 'True')


if __name__ == '__main__':
    main(sys.argv[1:])
