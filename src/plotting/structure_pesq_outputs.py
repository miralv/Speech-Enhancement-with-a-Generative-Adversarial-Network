import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.io.wavfile
import resampy
import os
from tools import find_rms, find_snr_factor
#import csv
""" Collect PESQ results in matrix form
"""
# Create table with the following columns: 
# speaker (0 or 1) file snr pesq_noisy pesq_enhanced
def read_pesq_results(filename_read, filename_save):
    header = "speaker file snr pesq_noisy pesq_enhanced\n"
    f = open(filename_save, mode='w')
    f.write(header)
    with open(filename_read,"r") as infile:
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
def findSpecificStats(file_name_read, snrs, num_each = 10,delim = ' '):
    """ Returns a dictionary on format
    noise:
    fraction improved
    pesq_enhanced
    pesq_noisy
    """
    mat = np.loadtxt(file_name_read, delimiter=delim, skiprows=1, usecols=[0,2,3,4])
    file_names = np.loadtxt(file_name_read, dtype = str, delimiter=delim, skiprows=1, usecols=[1])


    average_overall = np.zeros(2)
    average_overall[0] = np.mean(mat[:,2]) # noisy
    average_overall[1] = np.mean(mat[:,3]) # enhanced

    noise_stats = {}
    # reduce to only noise type
    noise_names = np.asarray(list(map(lambda x: x.split('_')[3], file_names)))
    noise_names_no_duplicates = np.asarray(list(dict.fromkeys(noise_names)))
    # need to remove n69 from statistics, as it was in the training set
    noise_remove = 'n69'
    ind_remove = noise_names_no_duplicates!=noise_remove
    noise_names_no_duplicates = noise_names_no_duplicates[ind_remove]

    indexes_improved = (mat[:,2]-mat[:,3])<0
    averages = np.zeros((2,len(snrs))) # for storing average snr score for noisy and enhanced speech (rad 0: noisy, rad 1: enhanced)

    for noise in noise_names_no_duplicates:
        indexes_this_noise = (noise_names == noise)
        indexes_improved_this_noise = indexes_this_noise * indexes_improved
        this_noise_result = mat[indexes_improved_this_noise,:]
        counts = np.zeros(len(snrs) + 1)
        pesq_scores_noisy = np.zeros(len(snrs))
        pesq_scores_enhanced = np.zeros(len(snrs))

        # snrs = [0,5,10,15] given as input, let the last element in counts store the mean
        # tot_each = this_noise_result.shape[0]/len(snrs)
        for i in range(this_noise_result.shape[0]):
            this_snr = this_noise_result[i,1]
            ind = snrs == this_snr
            counts[0:-1] +=ind
            #Pesq noisy
        
        # Here, we are interested in all results, not only improved
        this_noise_mat = mat[indexes_this_noise,:]
        for i in range(this_noise_mat.shape[0]):
            this_snr = this_noise_mat[i,1]
            ind = snrs == this_snr
            pesq_scores_noisy[ind] += this_noise_mat[i,2]
            pesq_scores_enhanced[ind] += this_noise_mat[i,3]
            averages[0,ind] += this_noise_mat[i,2]
            averages[1,ind] += this_noise_mat[i,3]

            #Pesq enhanced
        
        counts[-1] = np.mean(counts[0:-1])  
        # fractions_each = counts/tot_each
        # noise_stats[noise] = fractions_each
        # print(num_each)
        local_dict = {}
        local_dict['fractions_improved'] = counts/num_each
        local_dict['pesq_noisy'] = pesq_scores_noisy/num_each
        local_dict['pesq_enhanced'] = pesq_scores_enhanced/num_each
        noise_stats[noise] = local_dict
        print(noise)
        print(local_dict['fractions_improved'])
        print("\n\n")


    return noise_stats, averages/(num_each*len(noise_names_no_duplicates)), average_overall





plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
         ]
})

plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.labelsize'] = 14 #16 var for de minste figurene
plt.rcParams['font.size'] = 14#12
snrs = [0,5,10,15]


# # prøver 18 på det som er 3 i bredden

# #16 på de med to

# # running code
# file_read = "results/_pesq_results.txt"
# file_save = "pesq_results_no_z_table.csv"

# # read results of the files with no z

# read_pesq_results(file_read, file_save)
# snrs = [0,5,10,15]
# find_statistics(file_save,snrs,True)

# # Want to compare no_z with with_z
# file_plot_with_z = "pesq_results_table.csv"
# find_statistics(file_plot_with_z, snrs, True)


# """ Organize results from matlab scripts 
# """
# stoi_folder_no_z = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results.csv"
# pesq_matlab_folder_no_z = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results.csv"
# find_statistics(pesq_matlab_folder_no_z, snrs, True)



# """ Organize results from matlab script0 
# """
# snrs = [0,5,10,15]
# stoi_folder_no_z_longrun = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_no_z_16_may.csv"
# pesq_matlab_folder_no_z_longrun = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_no_z_16_may.csv"
# find_statistics(pesq_matlab_folder_no_z_longrun, snrs, True)

# find_sample_stats(file_name_read,epochs,"PESQ", snrs)


# """ Stats from samples"""
# epochs = np.arange(5.,41.,5.)
# file_name_read = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_samples_with_z_21_may.csv"
# def find_sample_stats(file_name_read,epochs, type_test, snrs, colors, save=False, savename="fig"):
#     reference_values = np.zeros(len(snrs))
#     if type_test == 'STOI':
#         reference_values = [0.929811, 0.966697, 0.985355, 0.994112]
#     elif type_test == 'PESQ':
#         reference_values = [1.643313, 1.896562, 2.244618, 2.645498]
#     else:
#         print("No valid test type. Choose PESQ or STOI.")
#         return 
#     mat = np.loadtxt(file_name_read, delimiter=' ', skiprows=1, usecols=[0,1,3,4])
#     print("Average %s score\n" % (type_test))
#     for snr in snrs: print(snr, end='')
#     print(" ")
#     epoch_scores = np.zeros((len(epochs),len(snrs)))
#     for i,epoch in enumerate(epochs):
#         print("Epcoch:%i" % (epoch), end='')
#         for j,snr in enumerate(snrs):
#             ind_ep = (mat[:,1] == epoch) * (mat[:,2] == snr)
#             epoch_scores[i,j] = np.mean(mat[ind_ep,3])
#             print(" %f" % ( epoch_scores[i,j]), end='')
#         print(" ")
#     legend_str = list(map(lambda x: str(x) + " dB", snrs))

#     fig,ax = plt.subplots(figsize = (8,6))
#     im = ax.plot(epochs,epoch_scores)
#     im2 = ax.hlines(reference_values, epochs[0], epochs[-1], linestyle='dashed', color=colors[0:len(reference_values)])
#     ax.legend(legend_str,title="SNR") #loc='upper right'
#     ax.set(xlabel="Epoch")
#     ax.set(ylabel=type_test)
#     ax.xaxis.set_ticks(epochs)
#     if type_test == 'STOI':
#         ax.set_ylim([np.min(epoch_scores), 1])
#     if type_test == 'PESQ':
#         ax.set_ylim([1.5, 3.0])

#     fig.tight_layout()


#     if save:
#         plt.savefig(savename)

#     # plt.show()
#     return epoch_scores


# """ Sammenlikn sample results and ordinary results for a run with and withoug adam"""



# adam_pesq = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_with_z_adam.csv"
# adam_stoi = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_with_z_adam.csv"
# adam_sample_pesq = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_samples_with_z_adam.csv"
# adam_sample_stoi = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_samples_with_z_adam.csv"


# find_sample_stats(adam_sample_pesq, epochs, "PESQ", snrs, colors)
# find_sample_stats(adam_sample_stoi, epochs, "STOI", snrs, colors)
# find_statistics(adam_pesq, snrs, True)
# find_statistics(adam_stoi, snrs, True)
# noise_stats_adam, averages_adam = findSpecificStats(adam_pesq, snrs)
# noise_stats_adam_stoi, averages_adam_stoi = findSpecificStats(adam_stoi, snrs)

# averages_adam
# averages_adam_stoi

# # old run, same config, but ooptimizer rmsprop
# stoi_folder_with_z = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_with_z_22_may.csv"
# pesq_matlab_folder_with_z = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_with_z_22_may.csv"
# find_statistics(pesq_matlab_folder_with_z, snrs, True)
# find_statistics(stoi_folder_with_z, snrs, True)
# find_sample_stats(stoi_folder_with_z,epochs,"STOI",snrs, colors)
# find_sample_stats(pesq_matlab_folder_with_z,epochs,"PESQ",snrs, colors)
# noise_stats,averages_pesq = findSpecificStats(pesq_matlab_folder_with_z,snrs)
# noise_stats_stoi,averages_stoi = findSpecificStats(stoi_folder_with_z,snrs)


# averages_pesq
# averages_stoi

# """********************************************************************************************************"""

# """Final training progress, with z, ~60 ep"""
# epochs = np.arange(5.,61.,5.)
# stoi_folder_with_z = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_final_with_z_60_ep.csv"
# pesq_matlab_folder_with_z = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_final_with_z_60_ep.csv"
# find_sample_stats(stoi_folder_with_z,epochs,"STOI",snrs, colors, True, "stoi_results_final_with_z_sample_60_ep.pdf")
# find_sample_stats(pesq_matlab_folder_with_z,epochs,"PESQ",snrs, colors, True, "pesq_results_final_with_z_sample_60_ep.pdf")


# # vil vite baseline score, i.e. hva som er tilhørende noisy score.


# #averages_pesq
# #averages_stoi

"""************************************************************************************************************"""
# Find reference lines to use in the sample plots.

reference_pesq_samples = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_sample_reference.csv"
reference_stoi_samples = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_sample_reference.csv"



find_sample_stats(reference_pesq_samples, [1], "PESQ", snrs, colors, False)
"""
Epcoch:1 1.643313 1.896562 2.244618 2.645498 
"""

find_sample_stats(reference_stoi_samples, [1], "STOI", snrs, colors, False)
"""
Epcoch:1 0.929811 0.966697 0.985355 0.994112
"""


"""************************************************************************************************************"""
# First results after NY , with z
n_hours_per_epoch = 200*40/3600
n_hours_per_epoch
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
epochs = np.arange(1.0,11.0,1.0)

stoi_folder_after_NY_samples_start = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_sample_after_NY_with_z_run_01.csv"
epoch_scores_with_1_stoi = find_sample_stats(stoi_folder_after_NY_samples_start, epochs, "STOI", snrs, colors, True,"stoi_sample_results_after_NY_with_z_run_1.pdf")

pesq_folder_after_NY_samples_start = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_sample_after_NY_with_z_run_01.csv"
epoch_scores_with_1_pesq = find_sample_stats(pesq_folder_after_NY_samples_start, epochs, "PESQ", snrs, colors, True,"pesq_sample_results_after_NY_with_z_run_1.pdf")



# Want to plot training progress, i.e. have a look at the training loss and validation loss.

G_file = "/home/shomec/m/miralv/Masteroppgave/Code/After_NY/with_z_run_01/G_20190605-113746"
training_mat_G = np.loadtxt(G_file, delimiter=' ', skiprows=2, usecols=[0,1,2])
val_mat_G = np.loadtxt(G_file, delimiter='|', skiprows=2, usecols=[1],dtype=str)
validation_mat_G = np.zeros((training_mat_G.shape))
for i in range(len(val_mat_G)):
    validation_mat_G[i,:]= val_mat_G[i].split(' ')[1:]

training_mat_G
validation_mat_G

# NB! det var en bug i utregningen av real og fake, => ikke noen vits i å plotte de. 
D_file = "/home/shomec/m/miralv/Masteroppgave/Code/After_NY/with_z_run_01/D_20190605-113746"
training_mat_D = np.loadtxt(D_file, delimiter=' ', skiprows=2, usecols=[0,1,2])
val_mat_D = np.loadtxt(D_file, delimiter='|', skiprows=2, usecols=[1],dtype=str) #husk: val mat real og val mat fake er feil
validation_mat_D = np.zeros((training_mat_D.shape))
for i in range(len(val_mat_D)):
    validation_mat_D[i,:]= val_mat_D[i].split(' ')[1:]

training_mat_D
validation_mat_D



fig,ax = plt.subplots(figsize=(8,5))
im = ax.plot(epochs, training_mat_D[:,0],label="Training loss")
ax.plot(epochs, validation_mat_D[:,0], label="Validation loss")
ax.set(ylabel=r"$V_D$",xlabel="Epoch")
ax.xaxis.set_ticks(np.arange(1,11,1))
plt.legend()
fig.tight_layout()
plt.savefig("V_D_sample_results_after_NY_with_z_run_1.pdf")
plt.show()



fig,ax = plt.subplots(figsize=(8,5))
im = ax.plot(epochs, training_mat_G[:,0],label="Training loss")
ax.plot(epochs, validation_mat_G[:,0], label="Validation loss")
ax.set(ylabel=r"$V_G$",xlabel="Epoch")
ax.xaxis.set_ticks(np.arange(1,11,1))
plt.legend()
fig.tight_layout()
plt.savefig("V_G_sample_results_after_NY_with_z_run_1.pdf")
plt.show()




fig,ax = plt.subplots(figsize=(8,5))
im = ax.plot(epochs, training_mat_G[:,2],label="Training loss")
ax.plot(epochs, validation_mat_G[:,2], label="Validation loss")
ax.set(ylabel=r"$\Vert x - \hat{x} \Vert_1$",xlabel="Epoch")
ax.xaxis.set_ticks(np.arange(1,11,1))
plt.legend()
fig.tight_layout()
plt.savefig("L1_sample_results_after_NY_with_z_run_1.pdf")
plt.show()


""" Start analyzing the results (test set)"""
stoi_folder_after_NY_with_z_run_1 = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_after_NY_with_z_run_01.csv"
pesq_folder_after_NY_with_z_run_1 = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_after_NY_with_z_run_01.csv"


find_statistics(pesq_folder_after_NY_with_z_run_1, snrs, True, "histogram_after_NY_with_z_run_1.pdf", delim=',')
find_statistics(stoi_folder_after_NY_with_z_run_1, snrs, True)
plt.show()

"plot histogram for each snr separately"
find_statistics(pesq_folder_after_NY_with_z_run_1, snrs, True, "histogram_after_NY_with_z_run_1_snr_0.pdf", True,   0, delim=',')
find_statistics(pesq_folder_after_NY_with_z_run_1, snrs, True, "histogram_after_NY_with_z_run_1_snr_5.pdf", True,   5, delim=',')
find_statistics(pesq_folder_after_NY_with_z_run_1, snrs, True, "histogram_after_NY_with_z_run_1_snr_10.pdf", True, 10, delim=',')
find_statistics(pesq_folder_after_NY_with_z_run_1, snrs, True, "histogram_after_NY_with_z_run_1_snr_15.pdf", True, 15, delim=',')



noise_stats, averages, avg_with_1 = findSpecificStats(pesq_folder_after_NY_with_z_run_1, snrs, num_each=10) #num each = number of different sentences
noise_stats_stoi, averages_stoi, avg_with_1_stoi = findSpecificStats(stoi_folder_after_NY_with_z_run_1, snrs, num_each=10) #num each = number of different sentences

averages
noise_stats
avg_with_1_stoi



save_file_noise_stats_with_z_stoi = "noise_stats_with_z_pesq_run_01.txt"
num_noises = 5
A = np.zeros((num_noises, len(snrs)*2))
print(A)
noisy_inds = np.arange(0, 2*len(snrs),2)
noisy_inds +1
for j,stats in enumerate(noise_stats_stoi):
    # print (stats)
    # print (noise_stats_with_2_stoi[stats]['pesq_noisy'])
    A[j,noisy_inds] = noise_stats_stoi[stats]['pesq_noisy']
    A[j,noisy_inds + 1] = noise_stats_stoi[stats]['pesq_enhanced']

print(A)

noise_stats

np.savetxt(save_file_noise_stats_with_z_stoi,A,fmt="%.2f")















"""************************************************************************************************************"""



def find_statistics(file_name_read, snrs, plots_wanted, savename="fig.pdf",only_one_snr=False, snr_chosen = 0, delim=' '):
    # Summer antall hvor det har skjedd en forbedring
    mat = np.loadtxt(file_name_read, delimiter=delim, skiprows=1, usecols=[0,2,3,4])
    # names = np.loadtxt(file_name_read,dtype='str',delimiter=' ',skiprows=1,usecols=1)
    improved = sum(mat[:,2]-mat[:,3]<0)
    # Percentage that has an improvement:
    print(" Fraction that has an improvement after enhancement: %f of %d files." %(improved/mat.shape[0], mat.shape[0]))
    #numpy.loadtxt(file, dtype=<class 'float'>, delimiter=' ', skiprows=1, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None)[source]
    indexes = (mat[:,2]-mat[:,3])<0
    #sorted = np.sort(names[indexes])
    improved_matrix = mat[indexes,:]

    # improved_matrix.shape
    counts = np.zeros(len(snrs))
    # snrs = [0,5,10,15] given as input
    tot_each = mat.shape[0]/4
    for i in range(improved_matrix.shape[0]):
        this_snr = improved_matrix[i,1]
        ind = snrs == this_snr
        counts +=ind

    fractions_each = counts/tot_each
    # Hva kjennetegner filene med improvement?
    # sum(improved_matrix[:,0]==1)
    # Want to analyze the pesq results. How? kunne ha løst dette med klasser.
    # counts
    print("SNR fraction of improved files:\n SNR  Fraction")
    for i in range(4):
        print("%d %f"% (snrs[i], fractions_each[i] ))
        # #plot the current file
        # #print(err)
        # err = np.asfarray(err)
        # np.savetxt("P16_t2.csv",err,delimiter=",")
        # #plt.loglog(h,err, label = r"$P=2, t=1$")

    # plt.xlabel(r"$h$")
    # plt.ylabel(r"$||e_h||_\infty$")
    # plt.legend(loc = "center right")
    # plt.show()
    if plots_wanted:
        if only_one_snr:
            ind_noisy = mat[:,1] == snr_chosen
            mat = mat[ind_noisy,:]
            print("Finding histogram for snr %i"%(snr_chosen))

        x_width = 0.1#0.125
        x = np.arange(-0.5,4.5 + x_width, x_width)
        number_in_each_bar_noisy = np.zeros(len(x)-1)
        number_in_each_bar_enhanced = np.zeros(len(x)-1)

        for i in range(mat.shape[0]):
            for j in range(1,len(x)):
                if  x[j-1] <= mat[i][2] and mat[i][2] < x[j]:
                    number_in_each_bar_noisy[j-1]+=1
                if x[j-1] <= mat[i][3] and mat[i][3] < x[j]:
                    number_in_each_bar_enhanced[j-1]+=1


        x_bar = np.arange(-0.5+x_width/2,4.5,x_width)
        number_in_each_bar_noisy
        print(np.sum(number_in_each_bar_enhanced))
        print(np.sum(number_in_each_bar_noisy))

        fig,ax = plt.subplots(figsize = (7,5))
        p1 = ax.bar(x_bar,number_in_each_bar_noisy/np.sum(number_in_each_bar_noisy), width=x_width,color='red',alpha=0.6)
        p2 = ax.bar(x_bar,number_in_each_bar_enhanced/np.sum(number_in_each_bar_enhanced), width=x_width, alpha=0.6)
        plt.legend((p1[0], p2[0]), ('Noisy', 'Enhanced'))
        ax.xaxis.set_ticks(np.arange(-0.5,5,0.5))
        ax.set(xlabel="PESQ")
        plt.savefig(savename)

        #plt.show()



"""************************************************************************************************************"""

# with z run 02

stoi_folder_after_NY_samples_with_second = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_sample_after_NY_with_z_run_02.csv"
epoch_scores_with_2_stoi = find_sample_stats(stoi_folder_after_NY_samples_with_second, epochs, "STOI", snrs, colors, True,"stoi_sample_results_after_NY_with_z_run_2.pdf")

pesq_folder_after_NY_samples_with_second = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_sample_after_NY_with_z_run_02.csv"
epoch_scores_with_2_pesq = find_sample_stats(pesq_folder_after_NY_samples_with_second, epochs, "PESQ", snrs, colors, True,"pesq_sample_results_after_NY_with_z_run_2.pdf")


stoi_folder_after_NY_with_z_run_2 = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_after_NY_with_z_run_02.csv"
pesq_folder_after_NY_with_z_run_2 = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_after_NY_with_z_run_02.csv"


find_statistics(pesq_folder_after_NY_with_z_run_2, snrs, True, "histogram_after_NY_without_z_run_1.pdf")
find_statistics(stoi_folder_after_NY_with_z_run_2, snrs, True)


find_statistics(pesq_folder_after_NY_with_z_run_2, snrs, True, "histogram_after_NY_with_z_run_2_snr_0.pdf", True, 0)
find_statistics(pesq_folder_after_NY_with_z_run_2, snrs, True, "histogram_after_NY_with_z_run_2_snr_5.pdf", True, 5)
find_statistics(pesq_folder_after_NY_with_z_run_2, snrs, True, "histogram_after_NY_with_z_run_2_snr_10.pdf", True, 10)
find_statistics(pesq_folder_after_NY_with_z_run_2, snrs, True, "histogram_after_NY_with_z_run_2_snr_15.pdf", True, 15)

plt.show()

noise_stats_with_2, averages_with_2, avg_with_2  = findSpecificStats(pesq_folder_after_NY_with_z_run_2, snrs, num_each=10) #num each = number of different sentences



averages_with_2
averages
averages_without

np.savetxt("save_averages.txt", averages_with_2, fmt="%.2f")



noise_stats_with_2_stoi, averages_with_2_stoi, avg_with_2_stoi = findSpecificStats(stoi_folder_after_NY_with_z_run_2, snrs, num_each=10) #num each = number of different sentences

averages_with_2_stoi
averages_stoi
np.savetxt("save_averages.txt", averages_with_2_stoi, fmt="%.2f")

avg_with_2_stoi


#  Get it on format for pasting it directly into the table generator
# that is: snr 0 5 10 15 \\ noisy & enhanced & noisy & enhanced osv

save_file_noise_stats_with_z = "noise_stats_with_z_stoi_run_02.txt"
num_noises = 5
A = np.zeros((num_noises, len(snrs)*2))
print(A)
noisy_inds = np.arange(0, 2*len(snrs),2)
noisy_inds +1
for j,stats in enumerate(noise_stats_with_2_stoi):
    # print (stats)
    # print (noise_stats_with_2_stoi[stats]['pesq_noisy'])
    A[j,noisy_inds] = noise_stats_with_2_stoi[stats]['pesq_noisy']
    A[j,noisy_inds + 1] = noise_stats_with_2_stoi[stats]['pesq_enhanced']

print(A)
noise_stats_with_2_stoi

np.savetxt(save_file_noise_stats_with_z,A,fmt="%.2f")


"""************************************************************************************************************"""


# with z run 03
stoi_folder_after_NY_samples_with_third = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_sample_after_NY_with_z_run_03.csv"
epoch_scores_with_3_stoi = find_sample_stats(stoi_folder_after_NY_samples_with_third, epochs, "STOI", snrs, colors, True,"stoi_sample_results_after_NY_with_z_run_3.pdf")

pesq_folder_after_NY_samples_with_third = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_sample_after_NY_with_z_run_03.csv"
epoch_scores_with_3_pesq = find_sample_stats(pesq_folder_after_NY_samples_with_third, epochs, "PESQ", snrs, colors, True,"pesq_sample_results_after_NY_with_z_run_3.pdf")

plt.show()
stoi_folder_after_NY_with_z_run_3 = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_after_NY_with_z_run_03.csv"
pesq_folder_after_NY_with_z_run_3 = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_after_NY_with_z_run_03.csv"


find_statistics(pesq_folder_after_NY_with_z_run_3, snrs, True, "histogram_after_NY_without_z_run_1.pdf")
find_statistics(stoi_folder_after_NY_with_z_run_3, snrs, True)

find_statistics(pesq_folder_after_NY_with_z_run_3, snrs, True, "histogram_after_NY_with_z_run_3_snr_0.pdf", True, 0)
find_statistics(pesq_folder_after_NY_with_z_run_3, snrs, True, "histogram_after_NY_with_z_run_3_snr_5.pdf", True, 5)
find_statistics(pesq_folder_after_NY_with_z_run_3, snrs, True, "histogram_after_NY_with_z_run_3_snr_10.pdf", True, 10)
find_statistics(pesq_folder_after_NY_with_z_run_3, snrs, True, "histogram_after_NY_with_z_run_3_snr_15.pdf", True, 15)



noise_stats_with_3, averages_with_3, avg_with_3 = findSpecificStats(pesq_folder_after_NY_with_z_run_3, snrs, num_each=10) #num each = number of different sentences
noise_stats_with_3_stoi, averages_with_3_stoi, avg_with_3_stoi = findSpecificStats(stoi_folder_after_NY_with_z_run_3, snrs, num_each=10) #num each = number of different sentences


avg_mean_pesq_with = (averages_with_3[1,:] + averages_with_2[1,:] + averages[1,:])/3.0
avg_mean_pesq_with

averages_without

avg_with_3_stoi


avg_mean_stoi_with = (averages_stoi[1,:] + averages_with_2_stoi[1,:] + averages_with_3_stoi[1,:])/3.0

avg_mean_stoi_with


np.savetxt("save_averages.txt", avg_mean_stoi_with, fmt="%.2f")


"""************************************************************************************************************"""



"""************************************************************************************************************"""
# Without z, run 1

stoi_folder_after_NY_samples_without_first = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_sample_after_NY_without_z_run_01.csv"
epoch_scores_without_1_stoi = find_sample_stats(stoi_folder_after_NY_samples_without_first, epochs, "STOI", snrs, colors, True,"stoi_sample_results_after_NY_without_z_run_1.pdf")

pesq_folder_after_NY_samples_without_first = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_sample_after_NY_without_z_run_01.csv"
epoch_scores_without_1_pesq = find_sample_stats(pesq_folder_after_NY_samples_without_first, epochs, "PESQ", snrs, colors, True,"pesq_sample_results_after_NY_without_z_run_1.pdf")


stoi_folder_after_NY_without_z_run_1 = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_after_NY_without_z_run_01.csv"
pesq_folder_after_NY_without_z_run_1 = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_after_NY_without_z_run_01.csv"


find_statistics(pesq_folder_after_NY_without_z_run_1, snrs, True, "histogram_after_NY_without_z_run_1.pdf")
find_statistics(stoi_folder_after_NY_without_z_run_1, snrs, True)


"plot histogram for each snr separately"
find_statistics(pesq_folder_after_NY_without_z_run_1, snrs, True, "histogram_after_NY_without_z_run_1_snr_0.pdf", True,   0, delim=',')
find_statistics(pesq_folder_after_NY_without_z_run_1, snrs, True, "histogram_after_NY_without_z_run_1_snr_5.pdf", True,   5, delim=',')
find_statistics(pesq_folder_after_NY_without_z_run_1, snrs, True, "histogram_after_NY_without_z_run_1_snr_10.pdf", True, 10, delim=',')
find_statistics(pesq_folder_after_NY_without_z_run_1, snrs, True, "histogram_after_NY_without_z_run_1_snr_15.pdf", True, 15, delim=',')

plt.show()

noise_stats_without, averages_without, avg_without_1 = findSpecificStats(pesq_folder_after_NY_without_z_run_1, snrs, num_each=10) #num each = number of different sentences
noise_stats_without_stoi, averages_without_stoi, avg_without_1_stoi = findSpecificStats(stoi_folder_after_NY_without_z_run_1, snrs, num_each=10) #num each = number of different sentences

averages_without

noise_stats


save_file_noise_stats_without_z_stoi = "noise_stats_without_z_stoi_run_01.txt"
num_noises = 5
A = np.zeros((num_noises, len(snrs)*2))
noisy_inds = np.arange(0, 2*len(snrs),2)
for j,stats in enumerate(noise_stats_without_stoi):
    # print (stats)
    # print (noise_stats_with_2_stoi[stats]['pesq_noisy'])
    A[j,noisy_inds] = noise_stats_without_stoi[stats]['pesq_noisy']
    A[j,noisy_inds + 1] = noise_stats_without_stoi[stats]['pesq_enhanced']

print(A)

noise_stats

np.savetxt(save_file_noise_stats_without_z_stoi,A,fmt="%.2f")











"""************************************************************************************************************"""

# Without z, run 2

stoi_folder_after_NY_samples_without_second = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_sample_after_NY_without_z_run_02.csv"
epoch_scores_without_2_stoi = find_sample_stats(stoi_folder_after_NY_samples_without_second, epochs, "STOI", snrs, colors, True,"stoi_sample_results_after_NY_without_z_run_2.pdf")

pesq_folder_after_NY_samples_without_second = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_sample_after_NY_without_z_run_02.csv"
epoch_scores_without_2_pesq = find_sample_stats(pesq_folder_after_NY_samples_without_second, epochs, "PESQ", snrs, colors, True,"pesq_sample_results_after_NY_without_z_run_2.pdf")


stoi_folder_after_NY_without_z_run_2 = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_after_NY_without_z_run_02.csv"
pesq_folder_after_NY_without_z_run_2 = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_after_NY_without_z_run_02.csv"


find_statistics(pesq_folder_after_NY_without_z_run_2, snrs, True, "histogram_after_NY_without_z_run_1.pdf")
find_statistics(stoi_folder_after_NY_without_z_run_2, snrs, True)


"plot histogram for each snr separately"
find_statistics(pesq_folder_after_NY_without_z_run_2, snrs, True, "histogram_after_NY_without_z_run_2_snr_0.pdf", True, 0)
find_statistics(pesq_folder_after_NY_without_z_run_2, snrs, True, "histogram_after_NY_without_z_run_2_snr_5.pdf", True, 5)
find_statistics(pesq_folder_after_NY_without_z_run_2, snrs, True, "histogram_after_NY_without_z_run_2_snr_10.pdf", True, 10)
find_statistics(pesq_folder_after_NY_without_z_run_2, snrs, True, "histogram_after_NY_without_z_run_2_snr_15.pdf", True, 15)



noise_stats_without_2, averages_without_2, avg_without_2 = findSpecificStats(pesq_folder_after_NY_without_z_run_2, snrs, num_each=10) #num each = number of different sentences
noise_stats_without_stoi_2, averages_without_stoi_2, avg_without_2_stoi = findSpecificStats(stoi_folder_after_NY_without_z_run_2, snrs, num_each=10) #num each = number of different sentences

averages_without_stoi
averages_without_stoi_2
np.savetxt("save_averages.txt", averages_without_stoi_2, fmt="%.2f")


"""************************************************************************************************************"""

# Without z, run 3

stoi_folder_after_NY_samples_without_third = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_sample_after_NY_without_z_run_03.csv"
epoch_scores_without_3_stoi = find_sample_stats(stoi_folder_after_NY_samples_without_third, epochs, "STOI", snrs, colors, True,"stoi_sample_results_after_NY_without_z_run_3.pdf")

pesq_folder_after_NY_samples_without_third = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_sample_after_NY_without_z_run_03.csv"
epoch_scores_without_3_pesq = find_sample_stats(pesq_folder_after_NY_samples_without_third, epochs, "PESQ", snrs, colors, True,"pesq_sample_results_after_NY_without_z_run_3.pdf")


stoi_folder_after_NY_without_z_run_3 = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_after_NY_without_z_run_03.csv"
pesq_folder_after_NY_without_z_run_3 = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_after_NY_without_z_run_03.csv"


find_statistics(pesq_folder_after_NY_without_z_run_3, snrs, True, "histogram_after_NY_without_z_run_1.pdf")
find_statistics(stoi_folder_after_NY_without_z_run_3, snrs, True)


"plot histogram for each snr separately"
find_statistics(pesq_folder_after_NY_without_z_run_3, snrs, True, "histogram_after_NY_without_z_run_3_snr_0.pdf", True, 0)
find_statistics(pesq_folder_after_NY_without_z_run_3, snrs, True, "histogram_after_NY_without_z_run_3_snr_5.pdf", True, 5)
find_statistics(pesq_folder_after_NY_without_z_run_3, snrs, True, "histogram_after_NY_without_z_run_3_snr_10.pdf", True, 10)
find_statistics(pesq_folder_after_NY_without_z_run_3, snrs, True, "histogram_after_NY_without_z_run_3_snr_15.pdf", True, 15)


noise_stats_without_3, averages_without_3, avg_without_3 = findSpecificStats(pesq_folder_after_NY_without_z_run_3, snrs, num_each=10) #num each = number of different sentences
noise_stats_without_stoi_3, averages_without_stoi_3, avg_without_3_stoi = findSpecificStats(stoi_folder_after_NY_without_z_run_3, snrs, num_each=10) #num each = number of different sentences

(avg_without_1[1] + avg_without_2[1] + avg_without_3[1])/3.0


(avg_without_2_stoi[1] +avg_without_3_stoi[1] + avg_without_1_stoi[1])/3.0

(avg_with_1[1] + avg_with_2[1] + avg_with_3[1])/3.0


(avg_with_2_stoi[1] +avg_with_3_stoi[1] + avg_with_1_stoi[1])/3.0


averages_without_stoi
averages_without_stoi_2
averages_without_stoi_3 # 1.20 1.38 1.60 1.79

avg_mean_stoi_without = (averages_without_stoi + averages_without_stoi_2 + averages_without_stoi_3)/3.0
np.savetxt("save_averages.txt", avg_mean_stoi_without, fmt="%.2f")

# mean, pesq 
# 1.20 1.36 1.56 1.73


"""************************************************************************************************************"""
epoch_scores_with_1_pesq



epoch_scores_with_all_pesq = np.zeros((3,len(epochs), len(snrs)))
epoch_scores_with_all_pesq[0] = epoch_scores_with_1_pesq
epoch_scores_with_all_pesq[1] = epoch_scores_with_2_pesq
epoch_scores_with_all_pesq[2] = epoch_scores_with_3_pesq
epoch_scores_with_all_pesq

epoch_scores_with_all_stoi = np.zeros((3,len(epochs), len(snrs)))
epoch_scores_with_all_stoi[0] = epoch_scores_with_1_stoi
epoch_scores_with_all_stoi[1] = epoch_scores_with_2_stoi
epoch_scores_with_all_stoi[2] = epoch_scores_with_3_stoi
epoch_scores_with_all_stoi


epoch_scores_without_all_pesq = np.zeros((3,len(epochs), len(snrs)))
epoch_scores_without_all_pesq[0] = epoch_scores_without_1_pesq
epoch_scores_without_all_pesq[1] = epoch_scores_without_2_pesq
epoch_scores_without_all_pesq[2] = epoch_scores_without_3_pesq
epoch_scores_without_all_pesq


epoch_scores_without_all_stoi = np.zeros((3,len(epochs), len(snrs)))
epoch_scores_without_all_stoi[0] = epoch_scores_without_1_stoi
epoch_scores_without_all_stoi[1] = epoch_scores_without_2_stoi
epoch_scores_without_all_stoi[2] = epoch_scores_without_3_stoi
epoch_scores_without_all_stoi


find_sample_stats_together(epoch_scores_with_all_pesq, epochs, "PESQ", snrs, colors, True, "pesq_sample_results_after_NY_with_z_all.pdf")
find_sample_stats_together(epoch_scores_without_all_pesq, epochs, "PESQ", snrs, colors, True, "pesq_sample_results_after_NY_without_z_all.pdf")
find_sample_stats_together(epoch_scores_with_all_stoi, epochs, "STOI", snrs, colors, True, "stoi_sample_results_after_NY_with_z_all.pdf")
find_sample_stats_together(epoch_scores_without_all_stoi, epochs, "STOI", snrs, colors, True, "stoi_sample_results_after_NY_without_z_all.pdf")


def find_sample_stats_together(epoch_scores,epochs, type_test, snrs, colors, save=False, savename="fig"):
    reference_values = np.zeros(len(snrs))
    if type_test == 'STOI':
        reference_values = [0.929811, 0.966697, 0.985355, 0.994112]
    elif type_test == 'PESQ':
        reference_values = [1.643313, 1.896562, 2.244618, 2.645498]
    else:
        print("No valid test type. Choose PESQ or STOI.")
        return 
    legend_str = list(map(lambda x: str(x) + " dB", snrs))
    fig,ax = plt.subplots(figsize = (8,8))
    for i in range(epoch_scores.shape[0]):
        print(i)
        for j in range(len(snrs)):
            im = ax.plot(epochs,epoch_scores[i,:,j], color = colors[j], alpha= 1 - i*0.2)

    ax.hlines(reference_values, epochs[0], epochs[-1], linestyle='dashed', color=colors[0:len(reference_values)])
    ax.legend(legend_str,title="SNR", loc='upper right')
    ax.set(xlabel="Epoch")
    ax.set(ylabel=type_test)
    ax.xaxis.set_ticks(epochs)
    if type_test == 'STOI':
        ax.set_ylim([0.78, 1])
    else:
        ax.set_ylim([1.5, 2.9])

    fig.tight_layout()
    if save:
        plt.savefig(savename)
    plt.show()


"""************************************************************************************************************"""
# New test, local
stoi_folder_after_NY_samples_with_local = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_sample_after_NY_with_local_z_run_04.csv"
epoch_scores_with_local_stoi = find_sample_stats(stoi_folder_after_NY_samples_with_local, epochs, "STOI", snrs, colors, True,"stoi_sample_results_after_NY_with_local_z_run_4.pdf")
plt.show()
pesq_folder_after_NY_samples_with_local = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_sample_after_NY_with_local_z_run_04.csv"
epoch_scores_with_local_pesq = find_sample_stats(pesq_folder_after_NY_samples_with_local, epochs, "PESQ", snrs, colors, True,"pesq_sample_results_after_NY_with_local_z_run_4.pdf")


stoi_folder_after_NY_with_local = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_after_NY_with_local_z_run_04.csv"
pesq_folder_after_NY_with_local = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_after_NY_with_local_z_run_04.csv"


find_statistics(pesq_folder_after_NY_with_local, snrs, True)
find_statistics(stoi_folder_after_NY_with_local, snrs, True)


noise_stats_with_local, averages_with_local, avg_with_local  = findSpecificStats(pesq_folder_after_NY_with_local, snrs, num_each=10) #num each = number of different sentences
noise_stats_with_local_stoi, averages_with_local_stoi, avg_with_local_stoi =  findSpecificStats(stoi_folder_after_NY

averages_with_local
avg_with_local
averages_with_local_stoi
avg_with_local_stoi

res_mat = np.zeros((2,5))
res_mat[0,0:-1]= averages_with_local[1]
res_mat[0,-1]= avg_with_local[1]
res_mat[1,0:-1]= averages_with_local_stoi[1]
res_mat[1,-1]= avg_with_local_stoi[1]

np.savetxt("save_averages.txt", res_mat, fmt="%.2f")

res_mat

#sjekk om den enhancer sammenlikningsverdig. hvis det, gjør tester (z, lokalt opptak med støy)
#### uten z

stoi_folder_after_NY_samples_without_local = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_sample_after_NY_without_local_z_run_04.csv"
epoch_scores_without_local_stoi = find_sample_stats(stoi_folder_after_NY_samples_without_local, epochs, "STOI", snrs, colors, True,"stoi_sample_results_after_NY_without_local_z_run_4.pdf")

pesq_folder_after_NY_samples_without_local = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_sample_after_NY_without_local_z_run_04.csv"
epoch_scores_without_local_pesq = find_sample_stats(pesq_folder_after_NY_samples_without_local, epochs, "PESQ", snrs, colors, True,"pesq_sample_results_after_NY_without_local_z_run_4.pdf")


stoi_folder_after_NY_without_local = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_after_NY_without_local_z_run_04.csv"
pesq_folder_after_NY_without_local = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_after_NY_without_local_z_run_04.csv"


find_statistics(pesq_folder_after_NY_without_local, snrs, True)
find_statistics(stoi_folder_after_NY_without_local, snrs, True)


noise_stats_without_local, averages_without_local, avg_without_local  = findSpecificStats(pesq_folder_after_NY_without_local, snrs, num_each=10) #num each = number of different sentences
noise_stats_without_local_stoi, averages_without_local_stoi, avg_without_local_stoi =  findSpecificStats(stoi_folder_after_NY_without_local, snrs, num_each=10)

averages_without_local
avg_without_local
averages_without_local_stoi
avg_without_local_stoi

res_mat = np.zeros((2,5))
res_mat[0,0:-1]= averages_without_local[1]
res_mat[0,-1]= avg_without_local[1]
res_mat[1,0:-1]= averages_without_local_stoi[1]
res_mat[1,-1]= avg_without_local_stoi[1]

np.savetxt("save_averages.txt", res_mat, fmt="%.2f")

res_mat
