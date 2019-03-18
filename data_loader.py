import scipy
import scipy.io.wavfile
from glob import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from tools import *

class DataLoader():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        #path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        cwd = os.getcwd()
        path = glob(cwd + "/pix2pix/datasets/" + '{}/{}/*'.format(self.dataset_name, data_type))


        batch_images = np.random.choice(path, size=batch_size)

        audios_A = []
        audios_B = []
        for audio_path in batch_images:
            audio = self.imread(audio_path)

            h, w, _ = audio.shape
            _w = int(w/2)
            audio_A, audio_B = audio[:, :_w, :], audio[:, _w:, :]

            audio_A = scipy.misc.imresize(audio_A, self.audio_res)
            audio_B = scipy.misc.imresize(audio_B, self.audio_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                audio_A = np.fliplr(audio_A)
                audio_B = np.fliplr(audio_B)

            audios_A.append(audio_A)
            audios_B.append(audio_B)

        audios_A = np.array(audios_A)/127.5 - 1.
        audios_B = np.array(audios_B)/127.5 - 1.

        return audios_A, audios_B

    def load_batch(self, batch_size=1, is_testing=False, SNRdB=5):
        data_type = "train" if not is_testing else "val"
        cwd = os.getcwd()
        # Collect all train or validation paths in the variable path
        # TODO: Change paths
        audio_paths = glob(cwd + "/pix2pix/datasets/" + '{}/{}/*'.format(self.dataset_name, data_type))
        #self.n_batches = int(len(path) / batch_size)
        noise_paths = []


        audios, noises = [],[]
        
        # TODO: Include sliding windows during training.
        while True:
            # 1 batch of Mixed, Clean
            audios_A, audios_B = [], []

            #batch = path[i*batch_size:(i+1)*batch_size]
            for audio_path in audio_paths:
                f_audio, audio_orig = scipy.io.wavfile.read(audio_path)
                # Downsample the audio to 16 kHz and scale it to [-1,1]
                audio = preprocess(audio_orig)
                audio = audio[0:( len(audio) - len(audio)%windowLength)]

                # Draw a random noise file
                noise_path = np.random.choice(noise_paths)
                f_noise, noise_orig = scipy.io.wavfile.read(noise_path)
                # Downsample the noise to 16 kHz and scale it to [-1,1]
                noise = downsample(noise_orig,noise=1)
                # Make the noise file have the same length as the audio file
                noise = extendVector(noise,len(audio))                
                            
                # Obtain desired SNR in mixed
                SNR_factor = findSNRfactor(audio,noise,SNRdB)
                mixed = audio + SNR_factor*noise

                # Rescale to obtiain values in [-1,1]
                max_val = np.max(np.abs(mixed))
                if max_val>1:
                    mixed = mixed/max_val

                audios.append(mixed)
                noises.append(clean)

            #audios_A = np.array(audios_A)/127.5 - 1.
            #audios_B = np.array(audios_B)/127.5 - 1.
            # Yields a list where audios_A is mixed audio
            # audios_B is corresponding clean audio

            # Draw a random audio file and a random part from it
            for (i in range(batch_size)):
                audioIndex = random.randint(0,len(audios))
                startIndex = random.randint(0,len(audios[audioIndex])-batch_size)

                audios_A.append(audio[audioIndex][startIndex:startIndex+batch_size])
                audios_B.append(noise[audioIndex][startIndex:startIndex+batch_size])
                
            yield audios_A, audios_B

