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


def analyse(obj, x_test, y_test):
    '''Analyse.'''
    # ## Mr N. Coder's wall
    n_to_show = 5000
    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]
    z_points = obj.encoder.predict(example_images)
    _plot_latent_space(z_points)

    min_x = min(z_points[:, 0])
    max_x = max(z_points[:, 0])
    min_y = min(z_points[:, 1])
    max_y = max(z_points[:, 1])

    # ### The new generated art exhibition
    figsize = 5

    plt.figure(figsize=(figsize, figsize))
    plt.scatter(z_points[:, 0], z_points[:, 1], c='black', alpha=0.5, s=2)

    grid_size = 10
    grid_depth = 3
    figsize = 15

    x = np.random.uniform(min_x, max_x, size=grid_size * grid_depth)
    y = np.random.uniform(min_y, max_y, size=grid_size * grid_depth)
    z_grid = np.array(list(zip(x, y)))
    reconst = obj.decoder.predict(z_grid)

    plt.scatter(z_grid[:, 0], z_grid[:, 1], c='red', alpha=1, s=20)
    plt.show()

    fig = plt.figure(figsize=(figsize, grid_depth))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(grid_size * grid_depth):
        ax = fig.add_subplot(grid_depth, grid_size, i + 1)
        ax.axis('off')
        ax.text(0.5, -0.35, str(np.round(z_grid[i], 1)),
                fontsize=10, ha='center', transform=ax.transAxes)

        ax.imshow(reconst[i, :, :, 0], cmap='Greys')

    n_to_show = 5000
    grid_size = 15
    figsize = 12

    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]
    example_labels = y_test[example_idx]

    z_points = obj.encoder.predict(example_images)

    plt.figure(figsize=(figsize, figsize))
    plt.scatter(z_points[:, 0], z_points[:, 1],
                cmap='rainbow', c=example_labels, alpha=0.5, s=2)
    plt.colorbar()
    plt.show()

    n_to_show = 5000
    grid_size = 20
    figsize = 8

    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]
    example_labels = y_test[example_idx]

    z_points = obj.encoder.predict(example_images)

    plt.figure(figsize=(5, 5))
    plt.scatter(z_points[:, 0], z_points[:, 1],
                cmap='rainbow', c=example_labels, alpha=0.5, s=2)
    plt.colorbar()

    x = np.linspace(min(z_points[:, 0]), max(z_points[:, 0]), grid_size)
    y = np.linspace(max(z_points[:, 1]), min(z_points[:, 1]), grid_size)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    z_grid = np.array(list(zip(xv, yv)))

    reconst = obj.decoder.predict(z_grid)

    plt.scatter(z_grid[:, 0], z_grid[:, 1], c='black', alpha=1, s=5)

    plt.show()

    fig = plt.figure(figsize=(figsize, figsize))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(grid_size**2):
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        ax.axis('off')
        ax.imshow(reconst[i, :, :, 0], cmap='Greys')


def _plot_reconstruct_orig(autoencoder, x_test):
    '''Plot reconstructed original paintings.'''
    n_to_show = 10
    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]

    z_points = autoencoder.encoder.predict(example_images)

    reconst_images = autoencoder.decoder.predict(z_points)

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(n_to_show):
        img = example_images[i].squeeze()
        ax = fig.add_subplot(2, n_to_show, i + 1)
        ax.axis('off')
        ax.text(0.5, -0.35, str(np.round(z_points[i], 1)),
                fontsize=10, ha='center', transform=ax.transAxes)
        ax.imshow(img, cmap='gray_r')

    for i in range(n_to_show):
        img = reconst_images[i].squeeze()
        ax = fig.add_subplot(2, n_to_show, i + n_to_show + 1)
        ax.axis('off')
        ax.imshow(img, cmap='gray_r')


def _plot_latent_space(z_points):
    '''Plot latent space.'''
    figsize = 12

    plt.figure(figsize=(figsize, figsize))
    plt.scatter(z_points[:, 0], z_points[:, 1], c='black', alpha=0.5, s=2)
    plt.show()
