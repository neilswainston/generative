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
    # Plot latent space from sample of encoded test data:
    z_data = get_latent_space(obj, x_data, k=5000)
    plot_latent_space(z_data)

    # Plot reconstructions:
    sample, recnst, z_points = get_reconstruct(obj, x_data, k=10)
    plot_reconstruct_orig(sample, recnst, z_points)

    # Plot continuum:
    sample, labels, z_points = get_continuum(obj, x_data, y_data, k=5000)
    z_grid = get_grid(z_points, grid_size=20)
    plot_continuum(obj, z_points, z_grid, labels)


def get_latent_space(obj, data, k):
    '''Get latent space from sample of data.'''
    sample = data[np.random.choice(data.shape[0], k, replace=False), :]
    return obj.encoder.predict(sample)


def get_reconstruct(obj, data, k):
    '''Get reconstructed from sample of data.'''
    sample = data[np.random.choice(data.shape[0], k, replace=False), :]
    z_points = obj.encoder.predict(sample)
    recnst = obj.decoder.predict(z_points)
    return sample, recnst, z_points


def get_continuum(obj, x_data, y_data, k):
    '''Get continuum of latent space.'''
    sample_idx = np.random.choice(range(len(x_data)), k)
    sample = x_data[sample_idx]
    labels = y_data[sample_idx]
    z_points = obj.encoder.predict(sample)
    return sample, labels, z_points


def get_grid(z_points, grid_size=20):
    '''Get grid.'''
    x = np.linspace(min(z_points[:, 0]), max(z_points[:, 0]), grid_size)
    y = np.linspace(max(z_points[:, 1]), min(z_points[:, 1]), grid_size)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    return np.array(list(zip(xv, yv)))


def plot_latent_space(z_points):
    '''Plot latent space.'''
    plt.figure()
    plt.scatter(z_points[:, 0], z_points[:, 1], c='black', alpha=0.5, s=2)
    plt.show()


def plot_reconstruct_orig(orig, recnst, z_points):
    '''Plot reconstructed original paintings.'''
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


def plot_continuum(obj, z_points, z_grid, sample_labels):
    '''Plot continuum grid over latent space.'''
    _plot_grid(z_points, z_grid, sample_labels)
    grid_size = int(len(z_grid)**0.5)

    reconst = obj.decoder.predict(z_grid)

    figsize = 8
    fig = plt.figure(figsize=(figsize, figsize))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(grid_size**2):
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        ax.axis('off')
        ax.imshow(reconst[i, :, :, 0], cmap='Greys')

    plt.show()


def _plot_grid(z_points, z_grid, labels):
    '''Plot grid over latent space.'''
    plt.figure(figsize=(5, 5))
    plt.scatter(z_points[:, 0], z_points[:, 1],
                cmap='rainbow', c=labels, alpha=0.5, s=2)
    plt.colorbar()
    plt.scatter(z_grid[:, 0], z_grid[:, 1], c='black', alpha=1, s=5)
    plt.show()
