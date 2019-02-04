import numpy as np
from scipy.signal import decimate
from tools import *
import librosa
import resampy

def preprocessing(rawAudio,q,N,windowLength,noise):
    """ Downsample, scale perform Hanning and apply FFT on the rawAudio

    # Arguments
        rawAudio: audio file
        q: downsampling factor
        N: audio file has values of type intN
        windowLength: number of samples per window
        noise: boolean variable, 1 if noise
        
    # Returns
        Preprocessed audio
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

    # Obtain windows and apply Hanning
    hanningArray = Hanning(y,windowLength)

    # Take the fourier transform, and get it returned in phormat z = x + iy
    fftArray = np.fft.fft(hanningArray)
    
    return fftArray


def Hanning(y,windowLength):
    """ Apply Hanning with 50 % overlap

    # Arguments
        y: audio array
        windowLength: number of samples per window
        
    # Returns
        Windowed audio
    """


    # Create a Hanning window
    window = np.hanning(windowLength)

    # Apply it
    # Remove entries s.t. there is an integer number of windows 
    n = len(y)-len(y)%windowLength
    y = y[0:n]
    nWindows = int(np.floor(2*n/windowLength)-1)
    hanningArray = np.zeros(shape=(nWindows,windowLength))
    for i in range(0,nWindows):
        startIndex = int(i * windowLength/2) # Assuming window length is dividible by two.
        hanningArray[i]= y[startIndex:startIndex+windowLength]*window

    return hanningArray