import scipy
import scipy.io.wavfile
from glob import glob
import numpy as np
import os
import random
#import matplotlib.pyplot as plt
from tools import *
from getData import getPaths

# Should it be a class or just a function?

def load_batch(options):
    """ Used for loading an epoch's random batches of data during training
    """

    audio_path = options['audio_path']
    noise_path = options['noise_path']
    batch_size = options['batch_size']
    n_batches = options['steps_per_epoch']
    snr_db = options['snr_db']
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
            audio = preprocess(audio_orig,f_audio)
            audio = audio[:(len(audio) - len(audio)%window_length)]

            f_noise, noise_orig = scipy.io.wavfile.read(noise_i)
            noise = preprocess(noise_orig, f_noise)
            noise = extendVector(noise, len(audio))


            # Obtain desired snr-level
            snr_factor = findSNRfactor(audio, noise, snr_db)
            mixed = audio + snr_factor*noise

            # Make sure that the values are still in [-1,1]
            max_val = np.max(np.abs(mixed))
            if max_val > 1:
                mixed = mixed/max_val
            
            # Draw a random part
            start_index = random.randint(0,len(mixed)-window_length)
            clean_audio_batch[j,:] = audio[start_index: start_index + window_length]
            mixed_audio_batch[j,:] = noise[start_index: start_index + window_length]

        # Yield a batch size of random samples with the wanted snr
        yield pre_emph(clean_audio_batch, pre_emph_const), pre_emph(mixed_audio_batch, pre_emph_const)


def prepare_test(options):
    """For a start, the test is just 1 audio clip, maybe with different noises?
    """
    audio_path = options['audio_path_test']
    noise_path = options['noise_path_test']
    snr_db = options['snr_db']
    window_length = options['window_length']
    z_dim = options['z_dim']
    pre_emph_const = options['pre_emph']

    f_audio, audio_orig = scipy.io.wavfile.read(audio_path)
    audio = preprocess(audio_orig,f_audio)
    audio = audio[:(len(audio) - len(audio)%window_length)]

    f_noise, noise_orig = scipy.io.wavfile.read(noise_path)
    noise = preprocess(noise_orig, f_noise)
    noise = extendVector(noise, len(audio))


    # Obtain desired snr-level
    snr_factor = findSNRfactor(audio, noise, snr_db)
    mixed = audio + snr_factor*noise

    # Make sure that the values are still in [-1,1]
    max_val = np.max(np.abs(mixed))
    if max_val > 1:
        mixed = mixed/max_val
    
    # Gir det mening å ha denne her, eller burde den ha ligget i main?
    n_windows = int(np.ceil(len(mixed)/window_length))
    z = np.random.normal(0,1,(n_windows,z_dim[0],z_dim[1]))
    
    #(0, 1, (batch_size, z_dim[0], z_dim[1]))

    # Slice to get it on format nwindows x windowlength
    audio = slice_vector(audio, options)
    mixed = slice_vector(mixed, options)


    return pre_emph(audio,pre_emph_const), pre_emph(mixed, pre_emph_const), z, max_val


