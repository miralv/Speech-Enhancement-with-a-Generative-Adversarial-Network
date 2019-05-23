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
plt.rcParams.update(plt.rcParamsDefault)

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
#plt.rcParams['axes.labelsize'] = 14 #16 var for de minste figurene
plt.rcParams['font.size'] = 12
# activation functions
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0.03 

plt.rcParams['axes.xmargin'] = 0.03
plt.rcParams['axes.ymargin'] = 0.03 


""" Plot activation functions"""
def relu(items):
    return list(map(lambda x: max(x,0),items))

def sigmoid(items):
    return list(map(lambda x: np.divide(1,1+np.exp(-x)),items))


def leakyrelu(items, alpha):
    return list(map(lambda x: max(0,x) + alpha* min(0,x), items))

def absoluterelu(items):
    return list(map(lambda x: max(0,x) -1* min(0,x), items))


x0 = -15
x1 = 15
x = np.arange(x0,x1,0.01)
    
y = relu(x)

fig,ax = plt.subplots(figsize = (10,5))
im = ax.plot(x,y)
ax.set(ylabel=r"$g(x)$",xlabel=r"$x$")
ax.xaxis.set_ticks(np.arange(x0,x1+1,5))
ax.yaxis.set_ticks(np.arange(0,x1+1,5))
plt.savefig('relu.pdf')


y = absoluterelu(x)
fig,ax = plt.subplots(figsize = (10,5))
im = ax.plot(x,y)
ax.set(ylabel=r"$g(x)$",xlabel=r"$x$")
ax.xaxis.set_ticks(np.arange(x0,x1+1,5))
ax.yaxis.set_ticks(np.arange(0,x1+1,5))
plt.savefig('absrelu.pdf')
plt.show()


y = absoluterelu(x)
fig,ax = plt.subplots(figsize = (10,5))
im = ax.plot(x,y,label=r'$\alpha=0.3$')
ax.set(ylabel=r"$g(x)$",xlabel=r"$x$")
ax.xaxis.set_ticks(np.arange(x0,x1+1,5))
ax.yaxis.set_ticks(np.arange(-5,x1+1,5))
ax.legend()
plt.savefig('leakyrelu.pdf')
plt.show()


z = sigmoid(x)
fig,ax = plt.subplots(figsize=(10,5))
im = ax.plot(x,z)
ax.set(ylabel=r"$g(x)$",xlabel=r"$x$")
ax.xaxis.set_ticks(np.arange(x0,x1+1,deltax=5))
#ax.yaxis.set_ticks(np.arange(0,x1+1,deltay=5))
plt.savefig('sigmoid.pdf')
plt.show()



""" Plot log(x) og log(1-x) """
x0 = 0
x1 = 1
x = np.arange(x0,x1,0.001)    
y1 = np.log(x)
y2 = np.log(1-x)
fig,ax = plt.subplots(figsize = (5,5))
im = ax.plot(x,y1,label=r'$\log(D(x))$')
im = ax.plot(x,y2,label=r'$\log(1-D(x))$')
ax.set(xlabel=r"$D(x)$")
ax.set_ylim([-7,1])
ax.legend(loc='upper right')
ax.xaxis.set_ticks(np.arange(x0,x1+1,deltax=0.2))
plt.savefig('logx.pdf')
plt.show()



def loss_d_lsgan(items, a, b):
    return list(map(lambda x: 0.5*np.power(x-b,2) + 0.5*np.power(x-a,2) ,items))

def loss_g_lsgan(items, c):
    return list(map(lambda x: 0.5*np.power(x-c,2),items))


def loss_g_gan(items):
    return np.log(1-items)


def loss_d_gan(items):
    return np.log(1-items) + np.log(items)




c =1
b = 1
a = 0
""" Plot the discriminators' loss functions"""
x0 = -15
x1 = 15
x = np.arange(x0,x1 + 0.01,0.01)    
d_lsgan = loss_d_lsgan(x,a,b) 
#d_gan = loss_d_gan(x)
fig,ax = plt.subplots(figsize = (5,5))
#im = ax.plot(x,d_gan,label=r'$D_{GAN}$')
im = ax.plot(x,d_lsgan,label=r'$D_{LSGAN}$')
ax.set(xlabel=r"$\hat{x}$")
#ax.set_ylim([-2,1])
#ax.legend()#loc='upper right')
#ax.xaxis.set_ticks(np.arange(x0,x1+1,deltax=0.2))
plt.savefig('loss_d.pdf')
plt.show()
