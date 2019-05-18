import scipy.io.wavfile
import resampy
#from tools import saveAudio
""" Test upsampling- do it have an effect?"""


orig_sr = 16000
new_sr = 48000
audio_i = "/home/shomec/m/miralv/Masteroppgave/Code/Deep-Learning-for-Speech-Separation/results_test_sample/epoch_30_enhanced_f2_4_x-c2161_TER_16k_ch01_snr_0.wav"
test_path = "/home/shomec/m/miralv/Masteroppgave/Code/Deep-Learning-for-Speech-Separation/results_test_sample/test_upsample/epoch_30_enhanced_f2_4_x-c2161_TER_16k_ch01_snr_0.wav"
f_audio, audio_orig = scipy.io.wavfile.read(audio_i)
yd = resampy.resample(audio_orig, orig_sr, new_sr)
saveAudio(yd, test_path, new_sr) 

