from __future__ import print_function, division
import scipy
import tensorflow

import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, PReLU, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam


def generator(options):
    """ A function that defines the generator based on the specified options.
    """

    skips = [] 
    num_layers = len(options['generator_encoder_num_kernels'])
    audio_shape = (options['window_length'], options['feat_dim'])
    generator_encoder_num_kernels = options['generator_encoder_num_kernels']
    generator_decoder_num_kernels = options['generator_decoder_num_kernels']
    filter_length = options['filter_length']
    strides = options['strides']
    padding = options['padding']
    use_bias = options['use_bias']
    show_summary = options['show_summary']



    ## Define the encoder
    encoder_in = Input(shape=audio_shape)
    encoder_out = encoder_in

    for layer_i, num_kernels in enumerate(generator_encoder_num_kernels):
        # Add convolution layer
        encoder_out = Conv1D(num_kernels, filter_length, strides=strides, padding=padding, use_bias=use_bias)(encoder_out)

        # Add skip connections
        if layer_i < num_layers -1:
            skips.append(encoder_out)

        # Apply PReLU
        encoder_out = PReLU(alpha_initializer='zeros', weights=None)(encoder_out)

    ## Define the intermediate noise layer z
    z_dim = options['z_dim']
    z = Input(shape=z_dim, name='noise_input')
    
    ## Define the decoder
    decoder_out = keras.layers.concatenate([encoder_out,z])
    # Shape variables updated through the loop
    n_rows = z_dim[0]
    n_cols = decoder_out.get_shape().as_list()[-1]

    for layer_i, num_kernels in enumerate(generator_decoder_num_kernels):
        shape_in = decoder_out.get_shape().as_list()

        # Need to transform the data to be in 3D, as conv2dtranspose need 3D input
        new_shape = (shape_in[1],1, shape_in[2])
        decoder_out = Reshape(new_shape)(decoder_out)   
        decoder_out = Conv2DTranspose(num_kernels, [filter_length, 1], strides=[strides, 1], padding=padding, use_bias=use_bias)(decoder_out)
        
        # Reshape back to 2D
        n_rows = strides*n_rows
        n_cols = num_kernels
        decoder_out.set_shape([None, n_rows, 1, n_cols])
        new_shape = (n_rows, n_cols)

        if layer_i == (num_layers-1):
            decoder_out = Reshape(new_shape, name='G_out')(decoder_out)
        else:
            decoder_out = Reshape(new_shape)(decoder_out)

        if layer_i < num_layers -1:
            # Apply PReLU
            decoder_out = PReLU(alpha_initializer='zeros', weights=None)(decoder_out)
            # Add skip connections
            skips_dec = skips[-(layer_i + 1)]
            decoder_out = keras.layers.concatenate([decoder_out,skips_dec])

        
    ## Create the model graph
    G = Model(inputs=[encoder_in,z], outputs=decoder_out)

    if show_summary:
        G.summary()

    return G




        







