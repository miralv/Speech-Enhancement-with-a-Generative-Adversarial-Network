import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob



# plot training and validation error 
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

g_files_with = glob.glob("/home/shomec/m/miralv/Masteroppgave/Code/After_NY/with_z_run_0*/G*")
len(g_files_with)
for g_file in g_files_with:
    plot_g_file(g_file)

d_files_with = glob.glob("/home/shomec/m/miralv/Masteroppgave/Code/After_NY/with_z_run_0*/D*")
len(g_files_with)
for d_file in d_files_with:
    plot_g_file(d_file)


g_files_without = glob.glob("/home/shomec/m/miralv/Masteroppgave/Code/After_NY/without_z_run_0*/G*")
for i, g_file in enumerate(g_files_without):
    save_g = g_files_without[i].split('/')[8] + "validation_error_G.pdf"
    plot_g_file(g_file, save_name = save_g)

d_files_without = glob.glob("/home/shomec/m/miralv/Masteroppgave/Code/After_NY/without_z_run_0*/D*")
for i, d_file in enumerate(d_files_without):
    save_d = d_files_without[i].split('/')[8] + "validation_error_D.pdf"
    plot_g_file(d_file, G=False, save_name = save_d)




epochs = np.arange(1.0,11.0,1.0)
def plot_g_file(g_file,epochs=np.arange(1.0,11.0,1.0), G=True, save_name="fig"):
    training_mat_G = np.loadtxt(g_file, delimiter=' ', skiprows=2, usecols=[0,1,2])
    val_mat_G = np.loadtxt(g_file, delimiter='|', skiprows=2, usecols=[1],dtype=str)
    validation_mat_G = np.zeros((training_mat_G.shape))
    for i in range(len(val_mat_G)):
        validation_mat_G[i,:]= val_mat_G[i].split(' ')[1:]


    fig,ax = plt.subplots(figsize=(8,5))
    im = ax.plot(epochs, training_mat_G[:,0],label="Training loss")
    ax.plot(epochs, validation_mat_G[:,0], label="Validation loss")
    if G:
        ax.set(ylabel=r"$V_G$",xlabel="Epoch")
    else:
        ax.set(ylabel=r"$V_D$",xlabel="Epoch")
        
    ax.xaxis.set_ticks(np.arange(1,11,1))
    plt.legend()
    fig.tight_layout()
    plt.savefig(save_name)
    



fig,ax = plt.subplots(figsize=(8,5))
im = ax.plot(epochs, training_mat_D[:,0],label="Training loss")
ax.plot(epochs, validation_mat_D[:,0], label="Validation loss")
ax.set(ylabel=r"$V_D$",xlabel="Epoch")
ax.xaxis.set_ticks(np.arange(1,11,1))
plt.legend()
fig.tight_layout()
#plt.savefig("V_D_sample_results_after_NY_with_z_run_1.pdf")
plt.show()



fig,ax = plt.subplots(figsize=(8,5))
im = ax.plot(epochs, training_mat_G[:,0],label="Training loss")
ax.plot(epochs, validation_mat_G[:,0], label="Validation loss")
ax.set(ylabel=r"$V_G$",xlabel="Epoch")
ax.xaxis.set_ticks(np.arange(1,11,1))
plt.legend()
fig.tight_layout()
#plt.savefig("V_G_sample_results_after_NY_with_z_run_1.pdf")
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

