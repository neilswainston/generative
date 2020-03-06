'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
import matplotlib.pyplot as plt
import numpy as np


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


def plot_orig_reconstruct(orig, recnst, z_data):
    '''Plot original versus reconstructed images.'''
    size = len(orig)

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for idx, (org, rec) in enumerate(zip(*[orig, recnst])):
        ax = fig.add_subplot(2, size, idx + 1)
        ax.axis('off')
        ax.imshow(org.squeeze(), cmap='gray_r')

        if z_data.shape[1] < 5:
            ax.text(0.5, -0.35, str(np.round(z_data[idx], 1)),
                    fontsize=10, ha='center', transform=ax.transAxes)

        ax = fig.add_subplot(2, size, idx + size + 1)
        ax.axis('off')
        ax.imshow(rec.squeeze(), cmap='gray_r')

    plt.show()
