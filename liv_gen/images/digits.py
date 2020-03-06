'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
from liv_gen.images import analysis, callbacks
from liv_gen.utils.data_utils import load_mnist
from liv_gen.utils.model_utils import ModelManager


def run(obj, build, out_dir='out'):
    '''Run.'''
    (x_train, _), (x_test, y_test) = load_mnist()

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

        obj.train(
            x_train,
            batch_size=32,
            epochs=10
        )

    analysis.analyse(obj, x_test, y_test)
