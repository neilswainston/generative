'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=wrong-import-order
import matplotlib.pyplot as plt
import numpy as np


def analyse(obj, x_data, y_data):
    '''Analyse.'''
    x_sample, y_sample, z_data = get_latent_space(obj, x_data, y_data, k=5000)

    # Plot latent space from sample of encoded test data:
    # plot_latent_space(z_data)

    # Plot original versus reconstructions:
    n = 16
    z_subsample = z_data[:n]
    recnst = obj.decoder.predict(z_subsample)
    plot_orig_reconstruct(x_sample[:n], recnst, z_subsample)

    # Plot continuum:
    z_grid = get_grid(z_data, grid_size=20)
    plot_latent_space(z_data, z_grid, y_sample)
    plot_reconstruct(obj, z_grid)


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


def plot_latent_space(z_points, z_selected=None, labels=None):
    '''Plot grid over latent space.'''
    plt.figure()

    if labels is not None:
        plt.scatter(z_points[:, 0], z_points[:, 1], cmap='rainbow', c=labels,
                    alpha=0.5, s=2)
        plt.colorbar()
    else:
        plt.scatter(z_points[:, 0], z_points[:, 1], cmap='rainbow',
                    alpha=0.5, s=2)

    if z_selected is not None:
        plt.scatter(z_selected[:, 0], z_selected[:, 1], c='black',
                    alpha=1, s=5)

    plt.show()


def plot_reconstruct(obj, z_data):
    '''Plot reconstructed images.'''
    grid_size = int(len(z_data)**0.5)

    reconst = obj.decoder.predict(z_data)

    figsize = 8
    fig = plt.figure(figsize=(figsize, figsize))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(grid_size**2):
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        ax.axis('off')
        ax.imshow(reconst[i, :, :, 0], cmap='Greys')

    plt.show()


def plot_orig_reconstruct(orig, recnst, z_points):
    '''Plot original versus reconstructed images.'''
    size = len(orig)

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for idx, (org, rec) in enumerate(zip(*[orig, recnst])):
        ax = fig.add_subplot(2, size, idx + 1)
        ax.axis('off')
        ax.imshow(org.squeeze(), cmap='gray_r')

        ax.text(0.5, -0.35, str(np.round(z_points[idx], 1)),
                fontsize=10, ha='center', transform=ax.transAxes)

        ax = fig.add_subplot(2, size, idx + size + 1)
        ax.axis('off')
        ax.imshow(rec.squeeze(), cmap='gray_r')
