import scipy
import scipy.io.wavfile
from glob import glob
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tools import *
from getData import getPaths

# Should it be a class or just a function?

def load_batch(options):
    audio_path = options['audio_path']
    noise_path = options['noise_path']
    batch_size = options['batch_size']
    n_epochs = options['n_epochs']
    snr_db = options['snr_db']
    window_length = options['window_length']

    audio_paths, noise_paths = getPaths(audio_path,noise_path)

    # TODO: Include sliding windows during training.


    for i in range(n_epochs):
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
        yield clean_audio_batch, mixed_audio_batch

