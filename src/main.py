from __future__ import print_function, division
import scipy
import tensorflow
from tensorflow.python.client import device_lib


import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout#, Concatenate
from keras.layers import BatchNormalization, Activation, PReLU, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

from discriminator import discriminator
from generator import generator
from data_loader import load_batch, prepare_test
from tools import *

def main():
    """
     Specify the specific structure of the discriminator and the generator,
     based on the architecture used in SEGAN.
    """
    # Parameters specified for the construction of the generator and discriminator
    options = {}
    options['Idun'] = True # Set to true when running on Idun, s.t. the audio path and noise path get correct
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
    options['pre_emph'] = 0.95

    # Some additional parameters needed in the training process
    if options['Idun']:
        options['audio_path'] = "/home/miralv/Master/Audio/sennheiser_1"
        options['noise_path'] = "/home/miralv/Master/Audio/Nonspeech"
    else:
        options['audio_path'] = "/home/shomec/m/miralv/Masteroppgave/Code/sennheiser_1"
        options['noise_path'] = "/home/shomec/m/miralv/Masteroppgave/Code/Nonspeech"



    options['batch_size'] = 20
    options['steps_per_epoch'] = 10
    options['n_epochs'] = 5
    options['snr_db'] = 5
    options['sample_rate'] = 16000

    print("Options are set.\n\n")

    # Print visible devices
    print("Print local devices:\n")
    print(device_lib.list_local_devices())
    print ("\n\n")


    ## Set up the individual models
    print("Setting up individual models\n")
    G = generator(options)
    print("G finished\n")
    D = discriminator(options)
    print("D finished\n\n")

    # Specify optimizer
    optimizer = Adam(lr=options['learning_rate'])


    # Compile the individual models
    print("Compile the individual models\n")
    D.compile(loss='mse', optimizer=optimizer)
    G.compile(loss='mae', optimizer=optimizer)



    ## Set up the combined model
    # TODO: Må de individuelle modellene kompileres i main?
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
    
    print("Set up the combined model\n")
    GAN = Model(inputs=[clean_audio_in, noisy_audio_in, z], outputs=[D_out, G_out])
    GAN.summary()
    #TODO: Check that the losses become correct with the model syntax
    GAN.compile(optimizer=optimizer,
                loss={'model_1': 'mae', 'model_2': 'mse'},
                loss_weights={'model_1': 100, 'model_2': 1})
    print(GAN.metrics_names)

    # # Tensorboard
    # if not os.path.exists("./logs"):
    #     os.makedirs("./logs")
    
    # log_path = "./logs"
    # callback = TensorBoard(log_path)
    # callback.set_model(GAN)
    # train_names = ['G_loss', 'G_adv_loss', 'G_l1Loss']
    
    ## Model training
    n_epochs = options['n_epochs']
    steps_per_epoch = options['steps_per_epoch']
    batch_size = options['batch_size']        


    start_time = datetime.datetime.now()
    # The real class labels for the discriminator inputs
    real_D = np.ones((batch_size, 1))  # For input pairs (clean, noisy)
    fake_D = np.zeros((batch_size, 1)) # For input pairs (enhanced, noisy)
    valid_G = np.array([1]*batch_size) # To compute the mse-loss


    print("Begin training\n")

    for epoch in range(1, n_epochs+1):
        for batch_i, (clean_audio, noisy_audio) in enumerate(load_batch(options)):
            ## Train discriminator
            # Get G's input in correct shape
            clean_audio = np.expand_dims(clean_audio, axis=2) #dim -> (batchsize,windowsize,1)
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
            [G_loss, G_D_loss, G_l1_loss] = GAN.train_on_batch(x=[clean_audio, noisy_audio, noise_input], y={'model_1': clean_audio, 'model_2': valid_G}) 

            # Print progress
            elapsed_time = datetime.datetime.now() - start_time
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [D real loss: %f] [D fake loss: %f] [G loss: %f] [G_D loss: %f] [G_L1 loss: %f] [Exec. time: %s]" % (epoch, n_epochs, batch_i + 1, steps_per_epoch, D_loss, D_loss_real, D_loss_fake, G_loss, G_D_loss, G_l1_loss, elapsed_time))


            # logs = [G_loss, G_D_loss, G_l1_loss]
            # write_log(callback, train_names, logs, epoch)



    # Want to plot training progress

    # Test the model 

    # # For now:
    # options['audio_path_test'] = "/home/shomec/m/miralv/Masteroppgave/Code/sennheiser_1/part_1/group_01/p1_g01_f1_1_t-a0001.wav"
    # options['noise_path_test'] = "/home/shomec/m/miralv/Masteroppgave/Code/Nonspeech/n1.wav"
    # clean,mixed,z,scaling_factor = prepare_test(options)

    # # Expand dims
    # # Need to get G's input in the correct shape
    # # First, get it into form (n_windows, window_length)
    # # Thereafter (n_windows, window_length,1)
    # clean = slice_vector(clean, options)
    # mixed = slice_vector(mixed, options)
    # audios_clean = np.expand_dims(clean, axis=2)
    # audios_mixed = np.expand_dims(mixed, axis=2)

    # # Condition on B and generate a translated version
    # G_out = G.predict([audios_mixed, z]) #Må jeg ha train = false?


    # # Postprocess = upscale from [-1,1] to int16
    # clean = postprocess(clean)
    # mixed = postprocess(mixed)
    # G_enhanced = postprocess(G_out,coeff = options['pre_emph'])

    # ## Save for listening
    # cwd = os.getcwd()
    # print(cwd)

    # if not os.path.exists("./results"):
    #     os.makedirs("./results")


    # # Want to save clean, enhanced and mixed.
    # if scaling_factor > 1:
    #     clean = np.divide(clean, scaling_factor)
    #     mixed = np.divide(mixed, scaling_factor)

    # sr = options['sample_rate']
    # path_audio = "./results/clean.wav"
    # path_noisy = "./results/noisy.wav"
    # path_enhanced = "./results/enhanced.wav"
    # saveAudio(clean, path_audio, sr)
    # saveAudio(mixed, path_noisy, sr)
    # saveAudio(G_enhanced, path_enhanced, sr)
    


    return 0


if __name__ == '__main__':
    res = main()