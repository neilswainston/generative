'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
import os.path

from keras.callbacks import ModelCheckpoint

from liv_gen.images import analysis, callbacks
from liv_gen.utils.callbacks import step_decay_schedule
from liv_gen.utils.data_utils import load_mnist
from liv_gen.utils.model_utils import ModelManager


def run(obj, build, data_dir='out'):
    '''Run.'''
    (x_train, _), (x_test, y_test) = load_mnist()

    manager = ModelManager(data_dir, obj.name)
    manager.init(obj, build)

    if build:
        obj.get_encoder().summary()
        obj.get_decoder().summary()

        obj.compile()

        obj.add_callbacks(get_callbacks(manager.get_folder(),
                                        print_batch=100,
                                        lr_decay=1,
                                        obj=obj))

        obj.train(
            x_train,
            batch_size=32,
            epochs=10
        )

    analysis.analyse(obj, x_test, y_test)


def get_callbacks(folder, print_batch, lr_decay, obj):
    '''Get callbacks.'''
    checkpoint1 = ModelCheckpoint(
        os.path.join(folder, 'weights/weights-{epoch:03d}.h5'),
        save_weights_only=True, verbose=1)

    checkpoint2 = ModelCheckpoint(
        os.path.join(folder, 'weights/weights.h5'),
        save_weights_only=True, verbose=1)

    lr_sched = step_decay_schedule(
        initial_lr=obj.learning_rate, decay_factor=lr_decay, step_size=1)

    image_callback = callbacks.ImageCallback(folder, print_batch, obj)

    return [checkpoint1, checkpoint2, lr_sched, image_callback]
