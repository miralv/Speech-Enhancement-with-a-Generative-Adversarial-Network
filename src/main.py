from __future__ import print_function, division
import scipy
import tensorflow
from tensorflow.python.client import device_lib
import glob


import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout#, Concatenate
from keras.layers import BatchNormalization, Activation, PReLU, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.models import model_from_json
import datetime
#import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
import sys
import numpy as np
import h5py
import os

from discriminator import discriminator
from generator import generator
from data_loader import load_batch, prepare_test
from tools import *


def test_audio(audio_path,path_save):
    """ Scale up enhanced audio
    """
    f_audio, audio_orig = scipy.io.wavfile.read(audio_path)
    # Looks like the mixed file usually have a maximum around 0.7 when scaled
    max_wanted = (2**15 -1)*0.7
    max_now = np.max(audio_orig)
    scale_factor =  max_wanted/max_now
    audio_scaled = audio_orig*scale_factor

    saveAudio(audio_scaled, path_save, sr=16000)





def main():
    """
     Specify the specific structure of the discriminator and the generator,
     based on the architecture used in SEGAN.
    """
    # test_path = "/home/shomec/m/miralv/Masteroppgave/Code/Deep-Learning-for-Speech-Separation/results/enhanced_n16.wav"
    # test_save = "/home/shomec/m/miralv/Masteroppgave/Code/Deep-Learning-for-Speech-Separation/results/enhanced_n16_v2.wav"
    # test_audio(test_path,test_save)


    # Need some flags too. (like, train, test, save load)
    TEST = True
    TRAIN = True
    SAVE = True
    LOAD = False

    # Parameters specified for the construction of the generator and discriminator
    options = {}
    options['Idun'] = False # Set to true when running on Idun, s.t. the audio path and noise path get correct
    # options['save_model'] = False
    # options['load_model'] = True
    options['window_length'] = 16384
    options['feat_dim'] = 1
    options['z_dim'] = (8, 1024) # Dimensions for the latent noise variable 
    options['filter_length'] = 31
    options['strides'] = 2
    options['padding'] = 'same'
    options['use_bias'] = True
    options['initializer_std_dev'] = 0.02
    options['generator_encoder_num_kernels'] = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
    options['generator_decoder_num_kernels'] = options['generator_encoder_num_kernels'][:-1][::-1] + [1]
    options['discriminator_num_kernels'] = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
    options['alpha'] = 0.3 # alpha in LeakyReLU
    options['show_summary'] = False
    options['learning_rate'] = 0.0002
    options['g_l1loss'] = 1000. # Just testing 
    options['pre_emph'] = 0.95

    # Training path
    if options['Idun']:
        options['audio_path'] = "/home/miralv/Master/Audio/sennheiser_1/part_1/Train"
        options['noise_path'] = "/home/miralv/Master/Audio/Nonspeech/Train"
    else:
        options['audio_path'] = "/home/shomec/m/miralv/Masteroppgave/Code/sennheiser_1/part_1/Train"
        options['noise_path'] = "/home/shomec/m/miralv/Masteroppgave/Code/Nonspeech_v2/Train" # /Train or /Validate or /Test



    options['batch_size'] = 32 #def:64
    options['steps_per_epoch'] = 20
    options['n_epochs'] = 40
    options['snr_db'] = 5
    options['snr_dbs_train'] = [0,10,15]
    options['snr_dbs_test'] = [0,5,10,15]
    options['sample_rate'] = 16000

    print("Options are set.\n\n")

    # # Print visible devices
    # print("Print local devices:\n")
    # print(device_lib.list_local_devices())
    # print ("\n\n")

    # Specify optimizer (Needed also if we choose not to train)
    # optimizer = Adam(lr=options['learning_rate'])
    # optimizer = keras.optimizers.RMSprop(lr=options['learning_rate'])

    # NB! i Segan er det definert to optimizere; en for d og en for g!!!
    optimizer_D = keras.optimizers.RMSprop(lr=options['learning_rate'])
    optimizer_G = keras.optimizers.RMSprop(lr=options['learning_rate'])

    if TRAIN:
        ## Set up the individual models
        print("Setting up individual models:\n")
        G = generator(options)
        print("G finished.\n")
        D = discriminator(options)
        print("D finished.\n\n")

        # Compile the individual models
        print("Compile the individual models.\n")
        D.compile(loss='mse', optimizer=optimizer_D)
        G.compile(loss='mae', optimizer=optimizer_G)


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
        
        print("Set up the combined model.\n")
        GAN = Model(inputs=[clean_audio_in, noisy_audio_in, z], outputs=[D_out, G_out])
        GAN.summary()
        #TODO: Check that the losses become correct with the model syntax
        GAN.compile(optimizer=optimizer_G,
                    loss={'model_1': 'mae', 'model_2': 'mse'},
                    loss_weights={'model_1': 500, 'model_2': 1})
        # print(GAN.metrics_names)

        # Tensorboard
        if not os.path.exists("./logs"):
            os.makedirs("./logs")
        
        # Write log manually for now
        log_file_path="./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        f = open(log_file_path,"w+")
        f.write("G_loss  G_D_loss  G_l1_loss\n")

        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        log_path = "./logs"
        callback = TensorBoard(log_path)
        callback.set_model(GAN)
        train_names = ['G_loss', 'G_adv_loss', 'G_l1Loss']
        
        ## Model training
        n_epochs = options['n_epochs']
        steps_per_epoch = options['steps_per_epoch']
        batch_size = options['batch_size']        


        start_time = datetime.datetime.now()
        # The real class labels for the discriminator inputs
        real_D = np.ones((batch_size, 1))  # For input pairs (clean, noisy)
        fake_D = np.zeros((batch_size, 1)) # For input pairs (enhanced, noisy)
        valid_G = np.array([1]*batch_size) # To compute the mse-loss

        print("Begin training.\n")

        for epoch in range(1, n_epochs+1):
            for batch_i, (clean_audio, noisy_audio) in enumerate(load_batch(options)):
                ## Train discriminator
                # Get G's input in correct shape
                clean_audio = np.expand_dims(clean_audio, axis=2) #dim -> (batchsize,windowsize,1)
                noisy_audio = np.expand_dims(noisy_audio, axis=2)

                # Har testet, Idun kommer seg hit. (men ikke lenger?)

                # Get G's enhanced audio
                noise_input = np.random.normal(0, 1, (batch_size, z_dim[0], z_dim[1])) #z
                G_enhanced = G.predict([noisy_audio, noise_input]) # Idea: Scale up enhanced output, since its magnitude generally is lower then sthe clean's magnitude
                # G_amp = findRMS(G_enhanced)
                # clean_amph = findRMS(clean_audio)
                # factor_try = findRMS(clean_audio)/findRMS(G_enhanced)

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

                # if (batch_i == (steps_per_epoch -1)):
                f.write("%f %f %f\n" % (G_loss, G_D_loss, G_l1_loss))
                logs = [G_loss, G_D_loss, G_l1_loss]
                write_log(callback, train_names, logs, epoch)

        f.close()
        print("Training finished.\n")

    # Want to plot training progress

    # Test the model 
    # Update testing. For now, update only for running locally.

    if TEST:
        if options['Idun']:
            options['audio_path_test'] = "/home/miralv/Master/Audio/sennheiser_1/part_1/Test/group_12/p1_g12_m1_1_t-a0001.wav"
            options['noise_path_test'] = "/home/miralv/Master/Audio/Nonspeech/Test"
        else:
            # options['audio_path_test'] = "/home/shomec/m/miralv/Masteroppgave/Code/sennheiser_1/part_1/Test/group_12/p1_g12_m1_1_t-a0001.wav"
            options['audio_folder_test'] = "/home/shomec/m/miralv/Masteroppgave/Code/sennheiser_1/part_1/Test/Selected"
            options['noise_folder_test'] = "/home/shomec/m/miralv/Masteroppgave/Code/Nonspeech_v2/Test" # /Train or /Validate or /Test

        print("Test the model on unseen noises and voices.\n\n")
        noise_list = glob.glob(options['noise_folder_test'] + "/*.wav")
        speech_list = glob.glob(options['audio_folder_test'] + "/*-c*.wav") # Want only the unique sentences

        if LOAD:
            print("Loading saved model\n")
            modeldir = os.getcwd()
            json_file = open(modeldir + "/Gmodel.json", "r")
            loaded_model_json = json_file.read()
            json_file.close()
            G = model_from_json(loaded_model_json)
            G.compile(loss='mean_squared_error', optimizer=optimizer_G)
            G.load_weights(modeldir + "/Gmodel.h5")


        SNR_dBs = options['snr_dbs_test']
        for speech_path in speech_list:
            options['audio_path_test'] = speech_path
            for noise_path in noise_list:
                options['noise_path_test'] = noise_path
                clean, mixed, z = prepare_test(options) #(snr_dbs, nwindows, windowlength)
                for i,snr_db in enumerate(SNR_dBs):
                    # Expand dims
                    # Need to get G's input in the correct shape
                    # First, get it into form (n_windows, window_length)
                    # Thereafter (n_windows, window_length,1)

                    # audios_clean = np.expand_dims(clean, axis=2)
                    audios_mixed = np.expand_dims(mixed[i], axis=2)

                    # Condition on B and generate a translated version
                    G_out = G.predict([audios_mixed, z[i]]) #meand [i,:,:]


                    # Postprocess = upscale from [-1,1] to int16
                    clean_res,_ = postprocess(clean[i,:,:], coeff = options['pre_emph'])
                    mixed_res,_ = postprocess(mixed[i,:,:], coeff = options['pre_emph'])
                    G_enhanced,_ = postprocess(G_out,coeff = options['pre_emph'])
                    # print("Was clean, mixed or enhanced scaled?")
                    # print("%f %f %f" % (scale_1, scale_2,g_scale))

                    ## Save for listening
                    cwd = os.getcwd()
                    #print(cwd)

                    if not os.path.exists("./results"):
                        os.makedirs("./results")

                    # Want to save clean, enhanced and mixed. 
                    sr = options['sample_rate']
                    # path_noisy = "./results/noisy_%s.wav" % (noise_path[-7:-4])
                    # path_enhanced = "./results/enhanced_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_%s.wav" % (noise_path[-7:-4])
                    if noise_path[-7]=='n':
                        path_enhanced = "./results/enhanced_%s_%s_snr_%d.wav" % (speech_path[-16:-4],noise_path[-7:-4], snr_db)# sentence id, noise id, snr_db
                        path_noisy = "./results/noisy_%s_%s_snr_%d.wav" % (speech_path[-16:-4],noise_path[-7:-4], snr_db)
                        path_clean = "./results/clean_%s_%s_snr_%d.wav" % (speech_path[-16:-4],noise_path[-7:-4], snr_db)

                    else:
                        path_enhanced = "./results/enhanced_%s_%s_snr_%d.wav" % (speech_path[-16:-4], noise_path[-16:-4], snr_db)
                        path_noisy = "./results/noisy_%s_%s_snr_%d.wav" % (speech_path[-16:-4], noise_path[-16:-4], snr_db)
                        path_clean = "./results/clean_%s_%s_snr_%d.wav" % (speech_path[-16:-4], noise_path[-16:-4], snr_db)

                    # Because pesq is testing corresponding clean, noisy and enhanced, must clean be stored similarly
                    saveAudio(clean_res, path_clean, sr) #per nå er det samme fil hver gang
                    saveAudio(mixed_res, path_noisy, sr)
                    saveAudio(G_enhanced, path_enhanced, sr)
    

    if SAVE and not LOAD:
        modeldir = cwd
        model_json = G.to_json()
        with open(modeldir + "/Gmodel.json", "w") as json_file:
            json_file.write(model_json)
        G.save_weights(modeldir + "/Gmodel.h5")
        print ("Model saved to " + modeldir)



if __name__ == '__main__':
    main()
