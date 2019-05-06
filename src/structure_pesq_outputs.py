import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io.wavfile
import resampy
import os
""" Collect PESQ results in matrix form
"""
file = "results/_pesq_results.txt"

# Create table with the following columns: 
# speaker (0 or 1) file snr pesq
header = "speaker file snr pesq\n"
f = open("pesq_results_table.csv", mode='w',)
f.write(header)
with open(file,"r") as infile:
    err = []
    counter = 1
    for i,line in enumerate(infile):
        if i != 0 and line!= ' \n':
            fields = line.split(" ")
            name = fields[1][:-5]
            pesq = float(fields[2])
            snr = float(name.split('_')[-1])
            speaker = 1 if name[9]=='f' else 0 # Logical variable representing speaker 1 or 0 (female and male)
            line_add = "%d %s %d %f\n" % (speaker, name, snr, pesq)
            f.write(line_add)

        # if ((counter%4)== 0):
        #     err.append(line.replace('\n','').split(":")[1])
        # counter+=1

f.close()
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
