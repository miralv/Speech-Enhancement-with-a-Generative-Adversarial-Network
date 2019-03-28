from __future__ import print_function, division
import scipy
import tensorflow

import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout#, Concatenate
from keras.layers import BatchNormalization, Activation, PReLU, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

from discriminator import discriminator
from generator import generator
from data_loader import load_batch

def main():
    """
     Specify the specific structure of the discriminator and the generator,
     based on the architecture used in SEGAN.
    """
    # Parameters specified for the construction of the generator and discriminator
    options = {}
    options['window_length'] = 16384
    options['feat_dim'] = 1
    options['z_dim'] = (8, 1024) # Dimensions for the latent noise variable 
    options['filter_length'] = 31
    options['strides'] = 2
    options['padding'] = 'same'
    options['use_bias'] = True
    options['generator_encoder_num_kernels'] = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
    options['generator_decoder_num_kernels'] = options['generator_encoder_num_kernels'][:-1][::-1] + [1]
    options['discriminator_num_kernels'] = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
    options['alpha'] = 0.3 # alpha in LeakyReLU
    options['show_summary'] = False
    options['learning_rate'] = 0.0002
    options['g_l1loss'] = 100.

    # Some additional parameters needed in the training process
    options['audio_path'] = "/home/shomec/m/miralv/Masteroppgave/Code/sennheiser_1"
    options['noise_path'] = "/home/shomec/m/miralv/Masteroppgave/Code/Nonspeech"
    options['batch_size'] = 20
    options['steps_per_epoch'] = 20
    options['n_epochs'] = 20 
    options['snr_db'] = 5
    



    ## Set up the individual models
    G = generator(options)
    D = discriminator(options)

    # Specify optimizer
    optimizer = Adam(lr=options['learning_rate'])


    # Compile the individual models
    D.compile(loss='mse', optimizer=optimizer)
    G.compile(loss='mae', optimizer=optimizer)



    ## Set up the combined model
    D.trainable = False
    audio_shape = (options['window_length'], options['feat_dim'])    
    z_dim = options['z_dim']
    # Prepare inputs
    clean_audio_in = Input(shape=audio_shape, name='in_clean')
    noisy_audio_in = Input(shape=audio_shape, name='in_noisy')
    z = Input(shape=z_dim, name='noise_input')
    # Prepare outputs
    G_out = G([noisy_audio_in, z])
    D_out = D([G_out, noisy_audio_in])

    GAN = Model(inputs=[clean_audio_in, noisy_audio_in, z], outputs=[D_out, G_out])
    GAN.summary()
    #TODO: Check that the losses become correct with the model syntax
    GAN.compile(optimizer=optimizer,
                loss={'model_1': 'mae', 'model_2': 'mse'},
                loss_weights={'model_1': 100, 'model_2': 1})
    print(GAN.metrics_names)


    
    ## Model training
    n_epochs = options['n_epochs']
    steps_per_epoch = options['steps_per_epoch']
    batch_size = options['batch_size']        


    start_time = datetime.datetime.now()
    # The real class labels for the discriminator inputs
    real_D = np.ones((batch_size, 1)) # For input pairs (clean, noisy)
    fake_D = np.zeros((batch_size, 1)) # For input pairs (enhanced, noisy)


    for epoch in range(1, n_epochs+1):
        for batch_i, (clean_audio, noisy_audio) in enumerate(load_batch(options)):
            ## Train discriminator
            # Get G's input in correct shape
            clean_audio = np.expand_dims(clean_audio, axis=2)
            noisy_audio = np.expand_dims(noisy_audio, axis=2)

            # Get G's enhanced audio
            noise_input = np.random.normal(0, 1, (batch_size, z_dim[0], z_dim[1])) #z
            G_enhanced = G.predict([noisy_audio, noise_input])

            # Comput the discriminator's loss
            D_loss_real = D.train_on_batch(x=[clean_audio, noisy_audio], y=real_D)
            D_loss_fake = D.train_on_batch(x=[G_enhanced, noisy_audio], y=fake_D)
            D_loss = np.add(D_loss_real, D_loss_fake)/2.0


            ## Train generator 
            # Keras expect a list of arrays > must reformat clean_audio
            [G_loss, G_D_loss, G_l1_loss] = GAN.train_on_batch(x=[clean_audio, noisy_audio, noise_input], y={'model_1': real_D, 'model_2': clean_audio[:,:,0]}) #usikker på siste outputparameter

            # Print progress
            elapsed_time = datetime.datetime.now() - start_time
            print("[Epoch %d/%d] [Batch %d/%d] [D real loss: %f] [D fake loss: %f] [G loss: %f] [G_D loss: %f] [G_L1 loss: %f] [Exec. time: %f]" % (epoch, n_epochs, batch_i, steps_per_epoch, D_loss_real, D_loss_fake, G_loss, G_D_loss, G_l1_loss, elapsed_time))
            
            
    return 0


if __name__ == '__main__':
    res = main()