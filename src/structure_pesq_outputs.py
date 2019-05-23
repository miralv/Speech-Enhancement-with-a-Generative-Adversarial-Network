import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.io.wavfile
import resampy
import os
from tools import findRMS, findSNRfactor
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

def find_statistics(file_name_read, snrs, plots_wanted):
    # Summer antall hvor det har skjedd en forbedring
    mat = np.loadtxt(file_name_read, delimiter=' ', skiprows=1, usecols=[0,2,3,4])
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
        x_width = 0.125
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
        plt.figure(1)
        p1 = plt.bar(x_bar,number_in_each_bar_noisy, width=x_width,color='red',alpha=0.6)
        #plt.show()
        p2 = plt.bar(x_bar,number_in_each_bar_enhanced, width=x_width, alpha=0.6)
        plt.legend((p1[0], p2[0]), ('Noisy', 'Enhanced'))
        plt.show()

def findSpecificStats(file_name_read, snrs, num_each = 10,delim = ' '):
    """ Returns a dictionary on format
    noise:
    fraction improved
    pesq_enhanced
    pesq_noisy
    """
    mat = np.loadtxt(file_name_read, delimiter=delim, skiprows=1, usecols=[0,2,3,4])
    file_names = np.loadtxt(file_name_read, dtype = str, delimiter=delim, skiprows=1, usecols=[1])
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

    return noise_stats, averages/(num_each*len(noise_names_no_duplicates))



# running code
file_read = "results/_pesq_results.txt"
file_save = "pesq_results_no_z_table.csv"

# read results of the files with no z

read_pesq_results(file_read, file_save)
snrs = [0,5,10,15]
find_statistics(file_save,snrs,True)

# Want to compare no_z with with_z
file_plot_with_z = "pesq_results_table.csv"
find_statistics(file_plot_with_z, snrs, True)


""" Organize results from matlab scripts 
"""
stoi_folder_no_z = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results.csv"
pesq_matlab_folder_no_z = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results.csv"
find_statistics(pesq_matlab_folder_no_z, snrs, True)





""" It would be interesting to gather the results according to noise type too.


Kan ha en egen tabell for alle demand-lydene.
Kan også skille en til en mellom feks car, station, and restaurant/cafeteria, park og hallway.
Guoning Hu:
vind, 
maskinlyd
vann
n16 og n17 representerer crowd noise. kan utelates, da det blir testet på filene fra demand.

Tabell:
                        SNRS
        PESQ (før vs etter)

vind                                    gjsnitt (eller gj.snittlig forbedring)
bil
crowd noise
station
park
gjsnitt med alle
kan ha en tilsvarende tabell med stoi


"""


""" Organize results from matlab script0 
"""
snrs = [0,5,10,15]
stoi_folder_no_z_longrun = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_no_z_16_may.csv"
pesq_matlab_folder_no_z_longrun = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_no_z_16_may.csv"
find_statistics(pesq_matlab_folder_no_z_longrun, snrs, True)

find_sample_stats(file_name_read,epochs,"PESQ", snrs)


""" Stats from samples"""
epochs = np.arange(5.,41.,5.)
file_name_read = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_samples_with_z_21_may.csv"
def find_sample_stats(file_name_read,epochs, type_test,snrs):
    mat = np.loadtxt(file_name_read, delimiter=' ', skiprows=1, usecols=[0,1,3,4])
    print("Average %s score\n" % (type_test))
    for snr in snrs: print(snr, end='')
    print(" ")
    epoch_scores = np.zeros((len(epochs),len(snrs)))
    for i,epoch in enumerate(epochs):
        print("Epcoch:%i" % (epoch), end='')
        for j,snr in enumerate(snrs):
            ind_ep = (mat[:,1] == epoch) * (mat[:,2] == snr)
            epoch_scores[i,j] = np.mean(mat[ind_ep,3])
            print(" %f" % ( epoch_scores[i,j]), end='')
        print(" ")
    plt.plot(epochs,epoch_scores)
    plt.legend(snrs)
    plt.xlabel("Epoch")
    plt.ylabel(type_test)
    plt.show()



""" Sammenlikn sample results and ordinary results for a run with and withoug adam"""



adam_pesq = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_with_z_adam.csv"
adam_stoi = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_with_z_adam.csv"
adam_sample_pesq = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_samples_with_z_adam.csv"
adam_sample_stoi = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_samples_with_z_adam.csv"


find_sample_stats(adam_sample_pesq, epochs, "PESQ", snrs)
find_sample_stats(adam_sample_stoi, epochs, "STOI", snrs)
find_statistics(adam_pesq, snrs, True)
find_statistics(adam_stoi, snrs, True)
findSpecificStats(adam_pesq, snrs)
findSpecificStats(adam_stoi, snrs)


# old run, same config, but ooptimizer rmsprop
stoi_folder_with_z = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/stoi_results_with_z_22_may.csv"
pesq_matlab_folder_with_z = "/home/shomec/m/miralv/Masteroppgave/Matlab_script/pesq_results_with_z_22_may.csv"
find_statistics(pesq_matlab_folder_with_z, snrs, True)
find_statistics(stoi_folder_with_z, snrs, True)
find_sample_stats(stoi_folder_with_z,epochs,"STOI",snrs)
find_sample_stats(pesq_matlab_folder_with_z,epochs,"PESQ",snrs)
noise_stats,averages_pesq = findSpecificStats(pesq_matlab_folder_with_z,snrs)
