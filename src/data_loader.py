import scipy
import scipy.io.wavfile
from glob import glob
import numpy as np
import os
import random
#import matplotlib.pyplot as plt
from tools import *
from getData import getPaths


def load_batch(options):
    """ Used for loading an epoch's random batches of data during training

    # Arguments
        options: specified in main()

    # Returns
        A batch of noise, speech and latent noise z

    """


    audio_path = options['audio_path']
    noise_path = options['noise_path']
    batch_size = options['batch_size']
    n_batches = options['steps_per_epoch']
    # snr_db = options['snr_db']
    snr_dbs = options['snr_dbs_train']
    window_length = options['window_length']
    pre_emph_const = options['pre_emph']

    # Get all paths in the training set
    audio_paths, noise_paths = getPaths(audio_path,noise_path)

    # TODO: Include sliding windows during training.


    for i in range(n_batches):
        # Extract randomly n=batch_size audio paths and noise paths
        audio_batch = np.random.choice(audio_paths, batch_size)
        noise_batch = np.random.choice(noise_paths, batch_size)

        # Variables to store the clean and noisy files
        clean_audio_batch = np.zeros((batch_size, window_length)) 
        mixed_audio_batch = np.zeros((batch_size, window_length))

        for j, (audio_i, noise_i) in enumerate(zip(audio_batch,noise_batch)):
            """ Read audio files,
            downsample to 16 kHz and scale to [-1,1] and
            make the noise file have the same length as the audio file.
            """

            f_audio, audio_orig = scipy.io.wavfile.read(audio_i)
            
            audio = preprocess_dataloader(audio_orig,f_audio)
            audio = audio[:(len(audio) - len(audio)%window_length)]

            f_noise, noise_orig = scipy.io.wavfile.read(noise_i)
            noise = preprocess_dataloader(noise_orig, f_noise)
            # Increase the possible extractions from noise
            if len(noise)< window_length:
                noise = extendVector(noise, 2*window_length)


            # Draw a random part from noise and audio and add them
            start_index_audio = random.randint(0,len(audio)-window_length)
            start_index_noise = random.randint(0,len(noise)-window_length)
            # Obtain desired snr-level
            snr_db = np.random.choice(snr_dbs)
            snr_factor = findSNRfactor(audio_orig, noise_orig, snr_db)
            # Scale and construct input windows
            clean_i = scaleDown(audio[start_index_audio: start_index_audio + window_length])
            mixed_i = clean_i + snr_factor*scaleDown(noise[start_index_noise: start_index_noise + window_length])

            # Make sure that the values are still in [-1,1]
            #TODO: Er det nødvendig å gjøre dette her? Holder det å gjøre det når lyden skal lyttes til? Altså for test sett reconstruction?
            max_val = np.max(np.abs(mixed_i))
            if max_val > 1:
                mixed_i = mixed_i/max_val
                clean_i = clean_i/max_val

            clean_audio_batch[j,:] = clean_i
            mixed_audio_batch[j,:] = mixed_i

            
        # Yield a batch size of random samples with the wanted snr
        yield pre_emph(clean_audio_batch, pre_emph_const), pre_emph(mixed_audio_batch, pre_emph_const)


def prepare_test(options):
    """ Used for loading the test set corresponding to given speech file and noise file, for all snr dbs wanted.

    # Arguments
        options: specified in main()

    # Returns
        The full test set for specified speech and noise file

    """

    audio_path = options['audio_path_test']
    noise_path = options['noise_path_test']
    snr_dbs = options['snr_dbs_test']
    window_length = options['window_length']
    z_dim = options['z_dim']
    pre_emph_const = options['pre_emph']

    f_audio, audio_orig = scipy.io.wavfile.read(audio_path)
    audio = preprocess(audio_orig,f_audio)
    audio = audio[:(len(audio) - len(audio)%window_length)]

    f_noise, noise_orig = scipy.io.wavfile.read(noise_path)
    noise = preprocess(noise_orig, f_noise)
    noise = extendVector(noise, len(audio))


    # mixed = np.zeros(shape =(len(snr_dbs),len(audio))) # Each row will contain mixed for corresponding snr
    # speech = np.zeros(shape =(len(snr_dbs),len(audio))) # Each row will contain mixed for corresponding snr

   # Prepare to get it on format len(snr_dbs) x nwindows x windowlength
    n_windows = int(np.ceil(len(audio)/window_length))
    speech_ready = np.zeros(shape=(len(snr_dbs), n_windows, window_length))
    mixed_ready = np.zeros(shape=(len(snr_dbs), n_windows, window_length))

    # Obtain desired snr-level
    # snr_factors = np.zeros((len(snr_dbs),1))
    for i,snr_db in enumerate(snr_dbs):
        snr_factor = findSNRfactor(audio_orig, noise_orig, snr_db)
        mixed = audio + snr_factor*noise

        # Make sure that the values are still in [-1,1]
        max_val = np.max(np.abs(mixed))
        if max_val < 1:
            max_val = 1.0
        mixed = mixed/max_val
        speech = audio/max_val

        # Slice vectors to get it on format nwindows x window length
        speech = slice_vector(speech, options)
        mixed = slice_vector(mixed, options)
        # Insert into output matrix
        speech_ready[i,:,:] = pre_emph(speech, pre_emph_const)
        mixed_ready[i,:,:] = pre_emph(mixed, pre_emph_const)



    #TODO: Gir det mening å ha denne her, eller burde den ha ligget i main?
    # Nå har z også snr-dbs på første aksen. deretter n_windows, z_dim0, z_dim1
    z = np.random.normal(0,1,(len(snr_dbs),n_windows,z_dim[0],z_dim[1]))
    #(0, 1, (batch_size, z_dim[0], z_dim[1]))

 
    return speech_ready, mixed_ready, z
