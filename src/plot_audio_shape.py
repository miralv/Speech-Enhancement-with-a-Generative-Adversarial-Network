import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io.wavfile
import resampy


audio_path = "/home/shomec/m/miralv/Masteroppgave/Code/sennheiser_1/part_1//Test/group_12/p1_g12_m1_1_t-a0001.wav"
noise_path = "/home/shomec/m/miralv/Masteroppgave/Code/Nonspeech_v2/Test/n77.wav" # /Train or /Validate or /Test

f_audio, audio_orig = scipy.io.wavfile.read(audio_path)
f_noise, noise_orig = scipy.io.wavfile.read(noise_path)


targetSr = 16000
audio_p = resampy.resample(audio_orig,f_audio,targetSr)
noise_p = resampy.resample(noise_orig,f_noise,targetSr)

len(noise_p)
len(audio_p)
45000

#save_path_audio = "/home/shomec/m/miralv/Masteroppgave/Code/figure_audio.pdf"
plt.figure(1,figsize=(20,5))
plt.plot(audio_p[25000:25000+40000],color='black')
plt.savefig("figure_audio.pdf",format="pdf")
#plt.show()

save_path_noise = "/home/shomec/m/miralv/Masteroppgave/Code/figure_noise.pdf"
plt.figure(figsize=(20,5))
plt.plot(noise_p[0:40000], color='black')
plt.savefig("figure_noise.pdf", format="pdf")

plt.show()
