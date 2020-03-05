'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
from digits_analysis import analyse
from models.vae import VariationalAutoencoder
from utils.loaders import load_model


# run params
SECTION = 'vae'
RUN_ID = '0002'
DATA_NAME = 'digits'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])


def main():
    '''main method.'''
    model = load_model(VariationalAutoencoder, RUN_FOLDER)
    analyse(model)


if __name__ == '__main__':
    main()
