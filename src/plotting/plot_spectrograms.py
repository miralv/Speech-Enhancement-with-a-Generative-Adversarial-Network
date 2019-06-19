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
import glob


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

# need to plot the spectrogram of the reconstructed speech also
# want all of the noise files to have equal rms
#noiseAudioFile ="C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Mixed/Simplified/enhancedMain508.12. try 1.wav"
n_noises = len(glob.glob("/home/shomec/m/miralv/Masteroppgave/Code/Nonspeech_v2/Validate/*.wav"))
# S_n_max = 6.3761314855842635
# S_n_min = -11.159604723845968

rms_wanted = 1000.0
S_n_max = 5.149918701962689
S_n_min = -10.37063967526063
for noiseAudioFile in glob.glob("/home/shomec/m/miralv/Masteroppgave/Code/Nonspeech_v2/Validate/*.wav"):
    f_noise, noise = scipy.io.wavfile.read(noiseAudioFile)
    noise_name = noiseAudioFile.split('/')[-1][:-4]
    save_name = "spectrogram_rms_1000_" + noise_name + ".pdf"
    #vil downsample til 16000 først
    f = 16000
    if f_noise != f:
        noise = resampy.resample(noise,f_noise,f)

    # kan la det være en makslengde på 4 sek = 4*16000 samples
    if len(noise)> 4 * f:
        noise = noise[0:4*f]
    rms =find_rms(noise)
    factor = rms_wanted/rms
    print("%s %f"% (noise_name, rms))
    f_n,t_n,S_n = signal.spectrogram(x=noise*factor, fs=f,window='hanning',nperseg=256,noverlap=128)
    plot_spectrogram(S_n_min,S_n_max,t_n,f_n,S_n,save_name)
    print("factor:%f" % (factor))

    """ code to find max and min used in the plots"""
    # cur_max = np.log10(np.float64(np.max(S_n)))
    # cur_min = np.log10(np.float64(np.min(S_n)))

    # if cur_max > S_n_max:
    #     S_n_max = cur_max
    # if cur_min < S_n_min:
    #     S_n_min = cur_min



S_n_max
S_n_min

def plot_spectrogram(minVal,maxVal,time,freq,spec,fileName):
    fig,ax = plt.subplots()
    im = ax.pcolormesh(time,freq,np.log(spec))
    ax.set(xlabel='Time [s]', ylabel='Frequency [Hz]')
    im.set_clim(minVal,maxVal)
    cbar = fig.colorbar(im)
    # cbar.set_ticks(np.arange(0,7,1))
    fig.tight_layout()
    plt.savefig(fileName)


# plotSpectrogram(S_n_min,S_n_max,t_n,f_n,S_n,'noiseSpectrogram.pdf')
