'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=wrong-import-order
from glob import glob
import os.path
import sys

from keras.preprocessing.image import ImageDataGenerator

from liv_gen.images import analysis, callbacks
from liv_gen.models.vae import VariationalAutoencoder
from liv_gen.utils.model_utils import ModelManager
import numpy as np


def run(build, out_dir='out', data_dir='./data/celeb/', batch_size=32):
    '''Run.'''
    obj = VariationalAutoencoder(
        name=os.path.splitext(os.path.basename(__file__))[0],
        input_dim=(128, 128, 3),
        encoder_conv_filters=[32, 64, 64, 64],
        encoder_conv_kernel_size=[3, 3, 3, 3],
        encoder_conv_strides=[2, 2, 2, 2],
        decoder_conv_t_filters=[64, 64, 32, 3],
        decoder_conv_t_kernel_size=[3, 3, 3, 3],
        decoder_conv_t_strides=[2, 2, 2, 2],
        z_dim=200,
        learning_rate=0.0005,
        r_loss_factor=10000,
        use_batch_norm=True,
        use_dropout=True)

    data_gen = ImageDataGenerator(rescale=1 / 255)

    data_flow = data_gen.flow_from_directory(
        data_dir,
        target_size=obj.input_dim[:2],
        batch_size=batch_size,
        shuffle=True,
        class_mode='input',
        subset='training'
    )

    num_images = len(np.array(glob(os.path.join(data_dir, '*/*.jpg'))))

    manager = ModelManager(out_dir, obj.name)
    manager.init(obj, build)

    if build:
        obj.get_encoder().summary()
        obj.get_decoder().summary()

        obj.compile()

        obj.add_callbacks(callbacks.get_callbacks(obj=obj,
                                                  folder=manager.get_folder(),
                                                  print_batch=100,
                                                  lr_decay=1))

        obj.train_with_generator(
            data_flow,
            epochs=10,
            steps_per_epoch=num_images / batch_size
        )

    # analysis.analyse(obj, x_test, y_test)


def main(args):
    '''main method.'''
    run(args[0] == 'True')


if __name__ == '__main__':
    main(sys.argv[1:])
