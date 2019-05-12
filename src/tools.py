import numpy as np
from scipy.signal import decimate
import scipy.io.wavfile
import resampy
import tensorflow as tf
#import librosa

def scaleDown(a, N=16):
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


def scaleUp(a, N=16):
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


def findSNRfactor(cleanAudio, noise, SNRdB):
    """ Find the SNR factor that noise must be multiplied by to obtain the specified
    SNRdB

    # Arguments
        cleanAudio: vector with the speech file
        noise: vector with the noise file
        SNRdB: wanted level of SNR given in dB
        
    # Returns
        The calculated factor
    """


    Anoise = findRMS(noise)
    if Anoise == 0:
        print('Dividing by zero!!')

    Aclean = findRMS(cleanAudio)
    ANoise_new = Aclean/(10**(SNRdB/20))
    factor = ANoise_new/Anoise
    return factor


def findRMS(vector):
    """ Fint the RMS of a vector.

    # Arguments
        vector: vector to calculate the RMS of     

    # Returns
        The calculated RMS
    """

    #Cast to a large dtype to prevent negative numbers due to overflow
    return np.sqrt(np.mean(np.power(vector,2,dtype='float64')))

def extendVector(vector, length):
    """ Extend the noise vector s.t. it achieves wanted length
    """

    while len(vector)<length:
        vector = np.append(vector,vector)

    vector = vector[:length]
    return vector


def preprocess(rawAudio, origSr):
    """ Downsample and scale.

    # Arguments
        rawAudio: audio file
        origSr: original sample rate

    # Returns
        Downsampled and scaled audio
    """

    # Target sample rate
    # TODO: listen to downsampled audio. check which filter is applied.

    targetSr = 16000
    yd = resampy.resample(rawAudio,origSr,targetSr)
    y = scaleDown(yd)

    return y

def preprocess_dataloader(rawAudio, origSr):
    """ Downsample and scale.

    # Arguments
        rawAudio: audio file
        origSr: original sample rate

    # Returns
        Downsampled audio
    """

    # Target sample rate
    # TODO: listen to downsampled audio. check which filter is applied.

    targetSr = 16000
    yd = resampy.resample(rawAudio,origSr,targetSr)

    return yd


    # # Downsample
    # if noise !=1:
    #     yd = decimate(rawAudio,q,ftype="fir")
    # else:
    #     origSr = 20000
    #     targetSr = 16000
    #     yd = resampy.resample(rawAudio, origSr, targetSr)
    # Shift to range [-1,1]



def slice_vector(vector, options, overlap=0.0):
    """
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

    #This is sufficient as long as overlap = 0.
    vectorized = np.reshape(audio, (-1))

    # De emph
    if coeff > 0:
        vectorized = de_emph(vectorized, coeff=coeff) 


    max_value = np.max(abs(vectorized))
    if (np.max(abs(audio))>1):
        vectorized = np.divide(vectorized,max_value)

    # Scale up
    recovered = scaleUp(vectorized)

    # Return the max_value such that the mixed and clean audio can be scaled accordingly.
    return recovered, max_value


def saveAudio(audio, path,sr):
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
    #diff = x[1:] - coeff*x[:-1]
    #concat = tf.concat(0,x0,diff)
    return x_pre_emph#concat



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


        