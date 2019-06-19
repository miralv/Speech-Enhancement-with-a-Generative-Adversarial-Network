import numpy as np
from scipy.signal import decimate
import scipy.io.wavfile
import resampy
import tensorflow as tf
import glob


def get_paths(speech_path,noise_path):
    """Returns audio paths and noise paths"""

    speech_paths = glob.glob(speech_path + "/*/p1_g*_" + "*.wav")       
    # Load noise files    
    noise_paths = glob.glob(noise_path + "/*.wav")

    return speech_paths,noise_paths


def scale_down(a, N=16):
    """ Scale down from intN to float in [-1,1]
    # N = 16 in all file types used

    # Arguments
        N: int type.  
        a: vector

    # Returns
        Downscaled vector    
    """


    c = np.divide(a, 2.0**(N-1) -1)    
    # Prevent values < -1
    d = list(map(lambda x: max(x,-1.0), c))
    return np.array(d,dtype = np.float64)


def scale_up(a, N=16):
    """ Scale up from  [-1,1] to intN,
    only int16 is used.

    # Arguments
        N: int type.  
        a: vector

    # Returns
        Upscaled vector    
    """


    b = list(map(lambda x : x*(2**(N-1)-1),a))
    if N == 32:
        return np.array(b,dtype= np.int32)
    if N==16: 
        return np.array(b,dtype = np.int16)
    return 0


def find_snr_factor(clean_speech, noise, SNRdB):
    """ Find the SNR factor that noise must be multiplied by to obtain the specified
    SNRdB

    # Arguments
        clean_speech: vector with the speech file
        noise: vector with the noise file
        SNRdB: wanted level of SNR given in dB
        
    # Returns
        The calculated factor
    """


    Anoise = find_rms(noise)
    if Anoise == 0:
        print('Dividing by zero!!')

    Aclean = find_rms(clean_speech)
    ANoise_new = Aclean/(10**(SNRdB/20))
    factor = ANoise_new/Anoise
    return factor


def find_rms(vector):
    """ Fint the RMS of a vector.

    # Arguments
        vector: vector to calculate the RMS of     

    # Returns
        The calculated RMS
    """

    #Cast to a large dtype to prevent negative numbers due to overflow
    return np.sqrt(np.mean(np.power(vector,2,dtype='float64')))

def extend_vector(vector, length):
    """ Extend the noise vector s.t. it achieves wanted length
    """

    while len(vector)<length:
        vector = np.append(vector,vector)

    vector = vector[:length]
    return vector


def preprocess(raw_audio, orig_sr):
    """ Downsample and scale.

    # Arguments
        raw_audio: audio file
        orig_sr: original sample rate

    # Returns
        Downsampled and scaled audio
    """

    # Target sample rate
    if orig_sr != 16000:
        target_sr = 16000
        yd = resampy.resample(raw_audio,orig_sr,target_sr)
    else:
        yd = raw_audio

    y = scale_down(yd)

    return y

def preprocess_dataloader(raw_audio, orig_sr):
    """ Downsample and scale.

    # Arguments
        raw_audio: audio file
        orig_sr: original sample rate

    # Returns
        Downsampled audio
    """

    # Target sample rate
    target_sr = 16000
    if target_sr != orig_sr:
        yd = resampy.resample(raw_audio,orig_sr,target_sr)
        return yd
    else:
        return raw_audio



def slice_vector(vector, options, overlap=0.0):
    """ Slice the vector
    """
    window_length = options['window_length']
    n_samples = len(vector)
    offset = int(window_length*overlap)
    min_length = offset
    if offset == 0:
        offset = window_length #no overlap

    # Initialize array
    sliced = np.array([]).reshape(0,window_length)

    for start_index in range(0,n_samples,offset):
        end_index = start_index + window_length

        if n_samples - start_index < min_length:
            break
        
        if end_index <= n_samples:
            slice_i = np.array([vector[start_index:end_index]])
        else:
            # Zero pad
            slice_i = np.concatenate((np.array([vector[start_index:]]), np.zeros((1, end_index - n_samples))), axis=1)

        sliced = np.concatenate((sliced, slice_i), axis=0)

    return sliced.astype('float64')


def postprocess(audio, coeff=0):
    """ Rescale and  map back to 1d
    """


    # Map back to 1d
    """rows,cols = audio.shape
    vectorized = np.zeros(shape=(1, rows*cols))
    index = 0
    for i in range(rows):
        # Fill one row at a time
        vectorized[index:index+cols] = audio[i,:]
        index += rows


    """

    #This is sufficient as long as overlap = 0 (which we have used)
    vectorized = np.reshape(audio, (-1))

    # De emph
    if coeff > 0:
        vectorized = de_emph(vectorized, coeff=coeff) 


    max_value = np.max(abs(vectorized))
    if (np.max(abs(audio))>1):
        vectorized = np.divide(vectorized,max_value)

    # Scale up
    recovered = scale_up(vectorized)

    # Return the max_value such that the mixed and clean audio can be scaled accordingly.
    return recovered, max_value


def save_audio(audio, path,sr):
    scipy.io.wavfile.write(path, sr, data=audio)


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()



def pre_emph(x, coeff=0.95):
    """
    Apply pre_emph on 2d data (batch size x window length)
    """
    #x0 = tf.reshape(x[0],[1,])
    x0 = x[:,0]
    x0 = np.expand_dims(x0, axis = 1)
    diff = x[:,1:] - coeff * x[:,:-1]
    x_pre_emph = np.concatenate((x0,diff),axis=1)
    return x_pre_emph



def de_emph(y, coeff=0.95):
    """
    Apply de_emph on test data: works only on 1d data
    """
    if coeff <= 0:
        return y

    x = np.zeros(shape = (y.shape[0] ,)) # Default is np.float64
    x[0] = y[0]
    for i in range(1, y.shape[0], 1):
        x[i] = coeff * x[i - 1] + y[i]
    
    return x


        