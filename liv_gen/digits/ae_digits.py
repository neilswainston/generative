'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
import os.path
import sys

from liv_gen.digits import analysis
from liv_gen.models.ae import Autoencoder
from liv_gen.utils.data_utils import load_mnist
from liv_gen.utils.model_utils import ModelManager


def run(build, data_dir='out'):
    '''Run.'''
    name = os.path.splitext(os.path.basename(__file__))[0]

    (x_train, _), (x_test, y_test) = load_mnist()

    obj = Autoencoder(
        name=name,
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

    manager = ModelManager(data_dir, name)
    manager.init(obj, build)

    if build:
        obj.get_encoder().summary()
        obj.get_decoder().summary()

        obj.compile()

        obj.train(
            x_train[:1000],
            batch_size=32,
            epochs=200,
            folder=manager.get_folder()
        )

    analysis.analyse(obj, x_test, y_test)


def main(args):
    '''main method.'''
    run(args[0] == 'True')


if __name__ == '__main__':
    main(sys.argv[1:])
