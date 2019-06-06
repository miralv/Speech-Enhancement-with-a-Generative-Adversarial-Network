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



#want to compare enhanced with clean
clean_file = "/home/shomec/m/miralv/Masteroppgave/Code/After_NY/testRun_new_setup/results/clean_f2_4_x-c0987_CAR_16k_ch01_snr_0.wav"
noisy_file = "/home/shomec/m/miralv/Masteroppgave/Code/After_NY/testRun_new_setup/results/noisy_f2_4_x-c0987_CAR_16k_ch01_snr_0.wav"
enhanced_file="/home/shomec/m/miralv/Masteroppgave/Code/After_NY/testRun_new_setup/results/enhanced_f2_4_x-c0987_CAR_16k_ch01_snr_0.wav"

_, clean = scipy.io.wavfile.read(clean_file)
_, noisy = scipy.io.wavfile.read(noisy_file)
_, enhanced = scipy.io.wavfile.read(enhanced_file)

rms_wanted = findRMS(clean)
rms_noisy = findRMS(noisy)
rms_enhanced = findRMS(enhanced)
factor_noisy = rms_wanted/rms_noisy
factor_enhanced = rms_wanted/rms_enhanced



plt.subplot(3,1,1)
plt.plot(clean)
plt.subplot(3,1,2)
plt.plot(noisy*factor_noisy)
plt.subplot(3,1,3)
plt.plot(enhanced*factor_enhanced)
plt.show()