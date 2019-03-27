import numpy as np
from scipy.signal import decimate
import resampy
#import librosa

def scaleDown(a,N=16):
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


def scaleUp(a,N=16):
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


def findSNRfactor(cleanAudio,noise,SNRdB):
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

def extendVector(vector,length):
    """ Extend the noise vector s.t. it achieves wanted length
    """

    while len(vector)<length:
        vector = np.append(vector,vector)

    vector = vector[:length]
    return vector


def preprocess(rawAudio, origSr):
    """ Downsample and scale

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

    # # Downsample
    # if noise !=1:
    #     yd = decimate(rawAudio,q,ftype="fir")
    # else:
    #     origSr = 20000
    #     targetSr = 16000
    #     yd = resampy.resample(rawAudio, origSr, targetSr)
    # Shift to range [-1,1]

