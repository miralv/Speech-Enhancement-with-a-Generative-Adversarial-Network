import numpy as np
from scipy.signal import decimate
import librosa
import resampy
def scaleDown(a,N):
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
    c = list(map(lambda x: max(x,-1.0), c))
    return c


def scaleUp(a,N):
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


    """ Calculate the ideal ratio mask

    # Arguments
        cleanAudioMatrix: matrix with preprocessed speech
        noise: matrix with preprocessed noise
        beta: tuning parameter
        
    # Returns
        The calculated ideal ratio mask
    """


    times, frequencies = noiseMatrix.shape
    IRM = np.zeros(shape = (times,frequencies))
    for t in range(0,times):
        for f in range(0,frequencies):
            #for each time-frequency unit
            speechEnergySquared = np.power(cleanAudioMatrix[t,f],2)
            noiseEnergySquared = np.power(noiseMatrix[t,f],2)
            IRM[t,f]= (speechEnergySquared/(speechEnergySquared + noiseEnergySquared))**beta
    return IRM


def extendVector(vector,length):
    """ Extend the noise vector s.t. it achieves wanted length
    """

    while len(vector)<length:
        vector = np.append(vector)

    vector = vector[:length]
    return vector


def preprocess(rawAudio,q=3,N=16,windowLength=256,noise=0):
    """ Downsample and scale

    # Arguments
        rawAudio: audio file
        q: downsampling factor
        N: audio file has values of type intN
        windowLength: number of samples per window
        noise: boolean variable, 1 if noise
        
    # Returns
        Downsampled and scaled audio
    """


    # Downsample
    if noise !=1:
        yd = decimate(rawAudio,q,ftype="fir")
    else:
        origSr = 20000
        targetSr = 16000
        yd = resampy.resample(rawAudio, origSr, targetSr)
    
    # Shift to range [-1,1]
    y = scaleDown(yd,N)

    return y
