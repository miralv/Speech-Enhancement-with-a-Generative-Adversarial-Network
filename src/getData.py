import numpy as np
import random
from pathlib import Path
import glob
import scipy.io.wavfile

def getAudio(audioPath, noisePath):
    """ Generate training data for use in the DNN


    # Arguments
        audioPath: path to folder where the speech files are located 
        noisePath: path to folder where the noise files are located
        
    # Returns
        audioFiles: vector with all audio files
        noiseFiles: vector with all noise files
    """

    # Want to have a nested list with one element per audio file
    
    #SHOULD WE DOWNSAMPLE HERE?
    #ANTAR JA!
    
    # Load clean audio
    audioFiles = []
    for i in range(1,10):
        #Inside group i 
        path = audioPath + "/part_1/group_0" + str(i) + "/p1_g0" + str(i) + "_m"
        for file in glob.glob(path + "*.wav"):
            _, audio = scipy.io.wavfile.read(file)
            audioFiles.append(audio)
       
    
    # Load noise files    
    noiseFiles = []
    for file in glob.glob(noisePath + "/*.wav"):
        _, noise = scipy.io.wavfile.read(file)
        noiseFiles.append(noise)
 

    return audioFiles, noiseFiles


def getPaths(audioPath,noisePath):
    """Returns audio paths and noise paths"""

    audioPaths = glob.glob(audioPath + "/*/p1_g*_m" + "*.wav")       
    # Load noise files    
    noisePaths = glob.glob(noisePath + "/*.wav")

    return audioPaths,noisePaths

