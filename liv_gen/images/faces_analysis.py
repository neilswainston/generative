'''
(c) University of Liverpool 2020

All rights reserved.

@author: neilswainston
'''
# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=wrong-import-order
import os

from scipy.stats import norm

from liv_gen.utils.data_utils import ImageLabelLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyse(vae):
    '''Analyse.'''
    data_folder = './data/celeb/'
    image_folder = './data/celeb/img_align_celeba/'

    # ## data
    input_dim = (128, 128, 3)
    att = pd.read_csv(os.path.join(data_folder, 'list_attr_celeba.csv'))
    image_loader = ImageLabelLoader(image_folder, input_dim[:2])
    att.head()

    # ## reconstructing faces
    batch_size = 10
    data_flow_generic = image_loader.build(att, batch_size=batch_size)

    example_batch = next(data_flow_generic)
    example_images = example_batch[0]

    z_points = vae.encoder.predict(example_images)

    reconst_images = vae.decoder.predict(z_points)

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(batch_size):
        img = example_images[i].squeeze()
        sub = fig.add_subplot(2, batch_size, i + 1)
        sub.axis('off')
        sub.imshow(img)

    for i in range(batch_size):
        img = reconst_images[i].squeeze()
        sub = fig.add_subplot(2, batch_size, i + batch_size + 1)
        sub.axis('off')
        sub.imshow(img)

    # ## Latent space distribution

    z_test = vae.encoder.predict_generator(
        data_flow_generic, steps=20, verbose=1)

    x = np.linspace(-3, 3, 100)

    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    for i in range(50):
        ax = fig.add_subplot(5, 10, i + 1)
        ax.hist(z_test[:, i], density=True, bins=20)
        ax.axis('off')
        ax.text(0.5, -0.35, str(i), fontsize=10,
                ha='center', transform=ax.transAxes)
        ax.plot(x, norm.pdf(x))

    plt.show()

    # ### Newly generated faces

    n_to_show = 30

    znew = np.random.normal(size=(n_to_show, vae.z_dim))

    reconst = vae.decoder.predict(np.array(znew))

    fig = plt.figure(figsize=(18, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(n_to_show):
        ax = fig.add_subplot(3, 10, i + 1)
        ax.imshow(reconst[i, :, :, :])
        ax.axis('off')

    plt.show()

    def get_vector_from_label(label, batch_size):
        '''Get vector from label.'''

        data_flow_label = image_loader.build(att, batch_size, label=label)

        # origin = np.zeros(shape=vae.z_dim, dtype='float32')
        current_sum_pos = np.zeros(shape=vae.z_dim, dtype='float32')
        current_n_pos = 0
        current_mean_pos = np.zeros(shape=vae.z_dim, dtype='float32')

        current_sum_neg = np.zeros(shape=vae.z_dim, dtype='float32')
        current_n_neg = 0
        current_mean_neg = np.zeros(shape=vae.z_dim, dtype='float32')

        current_vector = np.zeros(shape=vae.z_dim, dtype='float32')
        current_dist = 0

        print('label: ' + label)
        print('images : POS move : NEG move :distance : 𝛥 distance')
        while current_n_pos < 10000:

            batch = next(data_flow_label)
            im = batch[0]
            attribute = batch[1]

            z = vae.encoder.predict(np.array(im))

            z_pos = z[attribute == 1]
            z_neg = z[attribute == -1]

            if len(z_pos) > 0:
                current_sum_pos = current_sum_pos + np.sum(z_pos, axis=0)
                current_n_pos += len(z_pos)
                new_mean_pos = current_sum_pos / current_n_pos
                movement_pos = np.linalg.norm(new_mean_pos - current_mean_pos)

            if len(z_neg) > 0:
                current_sum_neg = current_sum_neg + np.sum(z_neg, axis=0)
                current_n_neg += len(z_neg)
                new_mean_neg = current_sum_neg / current_n_neg
                movement_neg = np.linalg.norm(new_mean_neg - current_mean_neg)

            current_vector = new_mean_pos - new_mean_neg
            new_dist = np.linalg.norm(current_vector)
            dist_change = new_dist - current_dist

            print(str(current_n_pos)
                  + '    : ' + str(np.round(movement_pos, 3))
                  + '    : ' + str(np.round(movement_neg, 3))
                  + '    : ' + str(np.round(new_dist, 3))
                  + '    : ' + str(np.round(dist_change, 3))
                  )

            current_mean_pos = np.copy(new_mean_pos)
            current_mean_neg = np.copy(new_mean_neg)
            current_dist = np.copy(new_dist)

            if np.sum([movement_pos, movement_neg]) < 0.08:
                current_vector = current_vector / current_dist
                print('Found the ' + label + ' vector')
                break

        return current_vector

    def add_vector_to_images(feature_vec):
        '''Add vector to images.'''

        n_to_show = 5
        factors = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

        example_batch = next(data_flow_generic)
        example_images = example_batch[0]
        # example_labels = example_batch[1]

        z_points = vae.encoder.predict(example_images)

        fig = plt.figure(figsize=(18, 10))

        counter = 1

        for i in range(n_to_show):

            img = example_images[i].squeeze()
            sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
            sub.axis('off')
            sub.imshow(img)

            counter += 1

            for factor in factors:

                changed_z_point = z_points[i] + feature_vec * factor
                changed_image = vae.decoder.predict(
                    np.array([changed_z_point]))[0]

                img = changed_image.squeeze()
                sub = fig.add_subplot(n_to_show, len(factors) + 1, counter)
                sub.axis('off')
                sub.imshow(img)

                counter += 1

        plt.show()

    batch_size = 500
    # attractive_vec = get_vector_from_label('Attractive', batch_size)
    # mouth_open_vec = get_vector_from_label('Mouth_Slightly_Open', batch_size)
    # smiling_vec = get_vector_from_label('Smiling', batch_size)
    # lipstick_vec = get_vector_from_label('Wearing_Lipstick', batch_size)
    # young_vec = get_vector_from_label('High_Cheekbones', batch_size)
    # male_vec = get_vector_from_label('Male', batch_size)
    # blonde_vec = get_vector_from_label('Blond_Hair', batch_size)

    eyeglasses_vec = get_vector_from_label('Eyeglasses', batch_size)

    # print('Attractive Vector')
    # add_vector_to_images(attractive_vec)

    # print('Mouth Open Vector')
    # add_vector_to_images(mouth_open_vec)

    # print('Smiling Vector')
    # add_vector_to_images(smiling_vec)

    # print('Lipstick Vector')
    # add_vector_to_images(lipstick_vec)

    # print('Young Vector')
    # add_vector_to_images(young_vec)

    # print('Male Vector')
    # add_vector_to_images(male_vec)

    print('Eyeglasses Vector')
    add_vector_to_images(eyeglasses_vec)

    # print('Blond Vector')
    # add_vector_to_images(blonde_vec)

    def morph_faces(start_image_file, end_image_file):
        '''Morph faces.'''

        factors = np.arange(0, 1, 0.1)

        att_specific = att[att['image_id'].isin(
            [start_image_file, end_image_file])]
        att_specific = att_specific.reset_index()
        data_flow_label = image_loader.build(att_specific, 2)

        example_batch = next(data_flow_label)
        example_images = example_batch[0]
        # example_labels = example_batch[1]

        z_points = vae.encoder.predict(example_images)

        fig = plt.figure(figsize=(18, 8))

        counter = 1

        img = example_images[0].squeeze()
        sub = fig.add_subplot(1, len(factors) + 2, counter)
        sub.axis('off')
        sub.imshow(img)

        counter += 1

        for factor in factors:

            changed_z_point = z_points[0] * (1 - factor) + z_points[1] * factor
            changed_image = vae.decoder.predict(np.array([changed_z_point]))[0]

            img = changed_image.squeeze()
            sub = fig.add_subplot(1, len(factors) + 2, counter)
            sub.axis('off')
            sub.imshow(img)

            counter += 1

        img = example_images[1].squeeze()
        sub = fig.add_subplot(1, len(factors) + 2, counter)
        sub.axis('off')
        sub.imshow(img)

        plt.show()

    start_image_file = '000238.jpg'
    end_image_file = '000193.jpg'  # glasses

    morph_faces(start_image_file, end_image_file)

    start_image_file = '000112.jpg'
    end_image_file = '000258.jpg'

    morph_faces(start_image_file, end_image_file)

    start_image_file = '000230.jpg'
    end_image_file = '000712.jpg'

    morph_faces(start_image_file, end_image_file)
