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

plt.rcParams['axes.labelsize'] = 14
#plt.rcParams['axes.labelsize'] = 14 #16 var for de minste figurene
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%",aspect='auto', pad="2%")
    return fig.colorbar(mappable, cax=cax)

#Plot spectrograms of clean, noise and mixed /home/shomec/m/miralv/Masteroppgave/Code/sennheiser_1/part_1/Test/Selected/p1_g12_m1_3_t-c1151.wav
cleanAudioFile ="/home/shomec/m/miralv/Masteroppgave/Code/sennheiser_1/part_1/Test/Selected/p1_g12_m1_3_t-c1151.wav"
#noiseAudioFile ="C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Mixed/Simplified/noiseScaled.wav"
mixedAudioFile ="/home/shomec/m/miralv/Masteroppgave/Code/After_NY/with_z_run_01/results/noisy_m1_3_t-c1151_n28_snr_0.wav"
enhanced_with_AudioFile ="/home/shomec/m/miralv/Masteroppgave/Code/After_NY/with_z_run_01/results/enhanced_m1_3_t-c1151_n28_snr_0.wav"
enhanced_without_AudioFile ="/home/shomec/m/miralv/Masteroppgave/Code/After_NY/without_z_run_01/results/enhanced_m1_3_t-c1151_n28_snr_0.wav"

#need to plot the spectrogram of the reconstructed speech also
#noiseAudioFile ="C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Mixed/Simplified/enhancedMain508.12. try 1.wav"

f_clean, clean = scipy.io.wavfile.read(cleanAudioFile)
# f_noise, noise = scipy.io.wavfile.read(noiseAudioFile)
f_mixed, mixed = scipy.io.wavfile.read(mixedAudioFile)
f_enhanced, enhanced = scipy.io.wavfile.read(enhanced_with_AudioFile)
f_enhanced, enhanced_without = scipy.io.wavfile.read(enhanced_without_AudioFile)

#vil downsample til 16000 først

f = 16000
clean = decimate(clean,int(f_clean/f),ftype="fir")
# clean = resampy.resample(clean,f_clean,f)

# vil kun ha første 6 sek
clean = clean[0:int(6*f)]
mixed = mixed[0:int(6*f)]
enhanced = enhanced[0:int(6*f)]
enhanced_without = enhanced_without[0:int(6*f)]

np.max(clean)

# clean = clean + np.ones(len(clean))
#vil ha samme lengde som mixed.

# l = mixed.shape[0]
# clean = clean[0:l]
# noise = noise[0:l]

f_c,t_c,S_c = signal.spectrogram(x=clean,fs=f)
# f_n,t_n,S_n = signal.spectrogram(x=noise, fs=f,window='hanning',nperseg=256,noverlap=128)
f_m,t_m,S_m = signal.spectrogram(x=mixed,fs=f)
f_e,t_e,S_e = signal.spectrogram(x=enhanced,fs=f)
f_e_w,t_e_w,S_e_w = signal.spectrogram(x=enhanced_without,fs=f)


z1 = np.max(S_c)
z1 = np.log10(np.float64(z1))
# z2 = np.max(S_n)
# z2 = np.log10(np.float64(z2))
z3 = np.max(S_m)
z3 = np.log10(np.float64(z3))
z4 = np.max(S_e)
z4 = np.log10(np.float64(z4))
z5 = np.max(S_e_w)
z5 = np.log10(np.float64(z5))

np.max(clean[0:16000])

# clean = decimate(clean, 3,ftype="fir")
# plt.specgram(clean,Fs=16000)
# plt.show()


# np.min(S_c)
# np.min(S_e)
# np.min(S_m)
maxVal = np.max([z1,z3,z4,x5])
minVal = np.log(0.0000001)
# minVal_c = np.min([np.log10(np.min(S_c))])
# minVal_m = np.min([np.log10(np.min(S_m))])
# minVal_e = np.min([np.log10(np.min(S_e))])
# minVal = np.min([minVal_c,minVal_e, minVal_m])
# minVal

def plotSpectrogram(minVal,maxVal,time,freq,spec,fileName):
    fig,ax = plt.subplots()
    im = ax.pcolormesh(time,freq,np.log(spec))
    ax.set(xlabel='Time [s]', ylabel='Frequency [Hz]')
    im.set_clim(minVal,maxVal)
    cbar = fig.colorbar(im)
    # cbar.set_ticks(np.arange(0,7,1))
    fig.tight_layout()
    plt.savefig(fileName)


plotSpectrogram(minVal,maxVal,t_c,f_c,S_c,'cleanAudioSpectrogram.pdf')
# plotSpectrogram(minVal,maxVal,t_n,f_n,S_n,'noiseSpectrogram.pdf')
plotSpectrogram(minVal,maxVal,t_m,f_m,S_m,'mixedSpectrogram.pdf')
plotSpectrogram(minVal,maxVal,t_e,f_e,S_e,'enhancedSpectrogram.pdf')
plotSpectrogram(minVal,maxVal,t_e_w,f_e_w,S_e_w,'enhancedSpectrogram_without_z.pdf')

S_c
f_c
t_c
minVal
maxVal

plt.show()