from __future__ import print_function, division
#import scipy
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
#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
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
    print("Hello world")
    # Parameters specified for the construction of the generator and discriminator
    # log_path = "./logs"
    # callback = TensorBoard(log_path)
    # callback.set_model(GAN)
    # train_names = ['G_loss', 'G_adv_loss', 'G_l1Loss']
    
    # ## Model training
    # n_epochs = options['n_epochs']
    # steps_per_epoch = options['steps_per_epoch']
    # batch_size = options['batch_size']        


    # start_time = datetime.datetime.now()
    # # The real class labels for the discriminator inputs
    # real_D = np.ones((batch_size, 1))  # For input pairs (clean, noisy)
    # fake_D = np.zeros((batch_size, 1)) # For input pairs (enhanced, noisy)
    # valid_G = np.array([1]*batch_size) # To compute the mse-loss

    # for epoch in range(1, n_epochs+1):
    #     for batch_i, (clean_audio, noisy_audio) in enumerate(load_batch(options)):
    #         ## Train discriminator
    #         # Get G's input in correct shape
    #         clean_audio = np.expand_dims(clean_audio, axis=2) #dim -> (batchsize,windowsize,1)
    #         noisy_audio = np.expand_dims(noisy_audio, axis=2)

    #         # Get G's enhanced audio
    #         noise_input = np.random.normal(0, 1, (batch_size, z_dim[0], z_dim[1])) #z
    #         G_enhanced = G.predict([noisy_audio, noise_input])

    #         # Comput the discriminator's loss
    #         D_loss_real = D.train_on_batch(x=[clean_audio, noisy_audio], y=real_D)
    #         D_loss_fake = D.train_on_batch(x=[G_enhanced, noisy_audio], y=fake_D)
    #         D_loss = np.add(D_loss_real, D_loss_fake)/2.0


    #         ## Train generator 
    #         # Keras expect a list of arrays > must reformat clean_audio
    #         [G_loss, G_D_loss, G_l1_loss] = GAN.train_on_batch(x=[clean_audio, noisy_audio, noise_input], y={'model_1': clean_audio, 'model_2': valid_G}) 

    #         # Print progress
    #         elapsed_time = datetime.datetime.now() - start_time
    #         print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [D real loss: %f] [D fake loss: %f] [G loss: %f] [G_D loss: %f] [G_L1 loss: %f] [Exec. time: %s]" % (epoch, n_epochs, batch_i + 1, steps_per_epoch, D_loss, D_loss_real, D_loss_fake, G_loss, G_D_loss, G_l1_loss, elapsed_time))


    #         logs = [G_loss, G_D_loss, G_l1_loss]
    #         write_log(callback, train_names, logs, epoch)



    # # Want to plot training progress

    # # Test the model 

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
    


    # return 0


if __name__ == '__main__':
    main()
