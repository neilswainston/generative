'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=wrong-import-order
from utils import analysis_utils, plot_utils


def analyse(obj, x_data, y_data):
    '''Analyse.'''
    x_sample, y_sample, z_data = \
        analysis_utils.get_latent_space(obj, x_data, y_data, k=5000)

    # Plot latent space from sample of encoded test data:
    # plot_utils.plot_latent_space(z_data)

    # Plot original versus reconstructions:
    n = 16
    z_subsample = z_data[:n]
    recnst = obj.decoder.predict(z_subsample)
    plot_utils.plot_orig_reconstruct(x_sample[:n], recnst, z_subsample)

    # Plot continuum:
    z_grid = analysis_utils.get_grid(z_data, grid_size=20)
    plot_utils.plot_latent_space(z_data, z_grid, y_sample)
    plot_utils.plot_reconstruct(obj, z_grid)
