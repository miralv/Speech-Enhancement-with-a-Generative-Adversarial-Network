import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io.wavfile
import resampy
import os
from tools import findRMS, findSNRfactor
#import csv
""" Collect PESQ results in matrix form
"""
file = "results/_pesq_results.txt"
# Create table with the following columns: 
# speaker (0 or 1) file snr pesq_noisy pesq_enhanced
header = "speaker file snr pesq_noisy pesq_enhanced\n"
f = open("pesq_results_table.csv", mode='w')
f.write(header)
with open(file,"r") as infile:
    err = []
    counter = 0
    counter_improved = 0
    for i,line in enumerate(infile,):
        if i != 0 and line!= ' \n':
            fields = line.split(" ")
            name = fields[1][6:-5]
            pesq = float(fields[2])
            snr = float(name.split('_')[-1])
            speaker = 1 if name[0]=='f' else 0 # Logical variable representing speaker 1 or 0 (female and male)
            if i%2: #Oddetallslinjer er noisy
                line_add = "%d %s %d %f" % (speaker, name, snr, pesq)
                f.write(line_add)
            else:
                line_add = " %f\n" % (pesq)
                f.write(line_add)


f.close()

to_matrix_file = "pesq_results_table.csv"
# Summer antall hvor det har skjedd en forbedring
mat = np.loadtxt(to_matrix_file, delimiter=' ', skiprows=1, usecols=[0,2,3,4])
names = np.loadtxt(to_matrix_file,dtype='str',delimiter=' ',skiprows=1,usecols=1)
mat.shape
improved = sum(mat[:,2]-mat[:,3]<0)
# Percentage that has an improvement:
improved/mat.shape[0]
#numpy.loadtxt(file, dtype=<class 'float'>, delimiter=' ', skiprows=1, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None)[source]
indexes = (mat[:,2]-mat[:,3])<0
sorted = np.sort(names[indexes])
improved_matrix = mat[indexes,:]

improved_matrix.shape
counts = np.zeros(4)
snrs = [0,5,10,15]
for i in range(improved_matrix.shape[0]):
    this_snr = improved_matrix[i,1]
    ind = snrs == this_snr
    counts +=ind
# Hva kjennetegner filene med improvement?
sum(improved_matrix[:,0]==1)
# Want to analyze the pesq results. How? kunne ha løst dette med klasser.

    # #plot the current file
    # #print(err)
    # err = np.asfarray(err)
    # np.savetxt("P16_t2.csv",err,delimiter=",")
    # #plt.loglog(h,err, label = r"$P=2, t=1$")

# plt.xlabel(r"$h$")
# plt.ylabel(r"$||e_h||_\infty$")
# plt.legend(loc = "center right")
# plt.show()














# plt.figure(1,figsize=(20,5))
# plt.plot(audio_p[25000:25000+40000],color='black')
# plt.savefig("figure_audio.pdf",format="pdf")
# #plt.show()

# save_path_noise = "/home/shomec/m/miralv/Masteroppgave/Code/figure_noise.pdf"
# plt.figure(figsize=(20,5))
# plt.plot(noise_p[0:40000], color='black')
# plt.savefig("figure_noise.pdf", format="pdf")

# plt.show()


# freq,noise = scipy.io.wavfile.read("/home/shomec/m/miralv/Masteroppgave/Code/Nonspeech_v2/Test/TCAR_16k_ch01.wav")
# rms_noise = findRMS(noise)
# np.sqrt(np.mean(np.power(noise,2,dtype='float64')))


# freq,clean = scipy.io.wavfile.read("/home/shomec/m/miralv/Masteroppgave/Code/sennheiser_1/part_1/Test/Selected/p1_g12_m1_3_t-c1982.wav")
# rms_clean = findRMS(clean)
# np.sqrt(np.mean(np.power(clean,2,dtype='float64')))


# snr_factor = findSNRfactor(clean,noise, 0)

# rms_noise*snr_factor
# rms_clean





