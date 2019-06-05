from scipy.signal import decimate
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
#import glob
import scipy.io.wavfile
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
import resampy
import librosa


#example code
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
         ]
})

plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['axes.labelsize'] = 14 #16 var for de minste figurene
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

#plt.figure(figsize=(4.5, 2.5))
#plt.plot(range(5))
#plt.text(2.5, 2.,size = '12')
#plt.xlabel(r"µ is not $\mu$")
#plt.tight_layout(.5)

##plt.savefig("pgf_texsystem.pdf")

##plt.plot([1, 2, 3, 4])
##plt.ylabel('some numbers')
#plt.show()



def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%",aspect='auto', pad="2%")
    return fig.colorbar(mappable, cax=cax)


#Plot spectrograms of the noise files in the test set
noiseAudioFile ="/home/shomec/m/miralv/Masteroppgave/Code/Nonspeech_v2/Test/STRAFFIC_16k_ch01.wav"

#need to plot the spectrogram of the reconstructed speech also
#noiseAudioFile ="C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Mixed/Simplified/enhancedMain508.12. try 1.wav"

f_noise, noise = scipy.io.wavfile.read(noiseAudioFile)

#vil downsample til 16000 først

f = 16000
clean = decimate(noise,int(f_noise/f),ftype="fir")
noise = resampy.resample(noise,f_noise,f)


f_n,t_n,S_n = signal.spectrogram(x=noise, fs=f,window='hanning',nperseg=256,noverlap=128)


z2 = np.max(S_n)
maxVal = np.max([z2])
minVal = 0

def plotSpectrogram(minVal,maxVal,time,freq,spec,fileName):
    fig,ax = plt.subplots()
    im = ax.pcolormesh(time,freq,np.log(spec))
    ax.set(xlabel='Time [s]', ylabel='Frequency [Hz]')
    #fig.tight_layout()
    im.set_clim(minVal,maxVal)
    cbar = fig.colorbar(im)
    cbar.set_ticks(np.arange(0,7,1))
    fig.tight_layout()
    plt.savefig(fileName)


plotSpectrogram(minVal,maxVal,t_n,f_n,S_n,'noiseSpectrogram.pdf')


#cbar.ax.get_yaxis().labelpad = 5
#cbar.ax.set_ylabel('log(Powerdensity))', rotation=270)

#plotter sigmoid 


for noise_file in noisefiles:
    plotSpectrogram(minVal, maxVal, t_n, f_n, S_n,'file_name.pdf')