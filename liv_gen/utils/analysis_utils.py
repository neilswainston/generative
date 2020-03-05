'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
import numpy as np


def get_latent_space(obj, x_data, y_data, k):
    '''Get latent space from sample of data.'''
    sample_idx = np.random.choice(range(len(x_data)), k)
    x_sample = x_data[sample_idx]
    y_sample = y_data[sample_idx]
    z_data = obj.encoder.predict(x_sample)
    return x_sample, y_sample, z_data


def get_grid(z_points, grid_size=20):
    '''Get grid.'''
    x = np.linspace(min(z_points[:, 0]), max(z_points[:, 0]), grid_size)
    y = np.linspace(max(z_points[:, 1]), min(z_points[:, 1]), grid_size)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    return np.array(list(zip(xv, yv)))
