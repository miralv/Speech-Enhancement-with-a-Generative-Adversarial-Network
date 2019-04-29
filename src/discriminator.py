from __future__ import print_function, division
import scipy
import tensorflow as tf

import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, PReLU, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam


def discriminator(options):
    """ A function that defines the discriminator based on the specified options.
    """
    
    audio_shape = (options['window_length'], options['feat_dim'])
    discriminator_num_kernels = options['discriminator_num_kernels']
    filter_length = options['filter_length']
    strides = options['strides']
    padding = options['padding']
    use_bias = options['use_bias']
    alpha = options['alpha']
    std_dev = options['initializer_std_dev']
    show_summary = options['show_summary']

    ## Define the discriminator's input and output
    # clean_audio_in = Input(shape=audio_shape, name='in_clean')
    # noisy_audio_in = Input(shape=audio_shape, name='in_noisy')
    clean_audio_in = Input(shape=audio_shape)
    noisy_audio_in = Input(shape=audio_shape)

    discriminator_out = keras.layers.concatenate([clean_audio_in,noisy_audio_in])


    for num_kernels in discriminator_num_kernels:
        # Add convolution layer
        discriminator_out = Conv1D(num_kernels, filter_length, strides=strides, padding=padding, use_bias=use_bias, init=tf.truncated_normal_initializer(stddev=std_dev))(discriminator_out)

        # Apply batch normalization
        discriminator_out = BatchNormalization()(discriminator_out)
        # Apply LeakyReLU
        discriminator_out = LeakyReLU(alpha=alpha)(discriminator_out)
    # discriminator_out = Conv1D(1, 1, padding=padding, use_bias=use_bias, name='logits_convolution')(discriminator_out)
    discriminator_out = Conv1D(1, 1, padding=padding, use_bias=use_bias, init=tf.truncated_normal_initializer(stddev=std_dev))(discriminator_out)
    discriminator_out = Flatten()(discriminator_out)
    # discriminator_out = Dense(1, activation='linear', name='D_output')(discriminator_out)
    discriminator_out = Dense(1, activation='linear')(discriminator_out)
    
    ## Construct model graph
    D = Model(inputs=[clean_audio_in, noisy_audio_in], outputs=discriminator_out)

    if show_summary:
        D.summary()

    return D


