import numpy as np
import random
from pathlib import Path
import glob
import scipy.io.wavfile

def getAudio():
    """ Generate training data for use in the DNN

    # Returns
        audioFiles: vector with all audio files
        noiseFiles: vector with all noise files
    """

    # Want to have a nested list with one element per audio file
    # Iterate through the different groups in Module 1
    
    # BURDE ALT DOWNSAMPLES OGSÅ MED EN GANG?

    # Load clean audio
    audioFiles = []
    for i in range(1,13):
        #Inside group i 
        filename = "./sennheiser_1/part_1/group_" + str(i) + "/p1_g" + str(i) + "_m"
        for file in glob.glob(filename + "*.wav"):
            _, audio = scipy.io.wavfile.read(file)
            audioFiles.append(audio)
       
    
    # Load noise files    
    #noiseFiles = []
    #for file in glob.glob(noiseFolder + "*.wav"):
    #    _, noise = scipy.io.wavfile.read(file)
    #    noiseFiles.append(noise)
 

    return audioFiles
