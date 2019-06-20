# Speech Enhancement with a Generative Adversarial Network

This project aims to enhance speech with a generative adversarial network. 
It is based on a two-player game between a generator and a discriminator - the generator learns to map from noisy to cleaner speech through a competition with a discriminator.

We used speech from the Norwegian speech database [NB Tale](https://www.nb.no/sprakbanken/show?serial=sbr-31&lang=nn) and noise files from the noise databases [Demand](https://zenodo.org/record/1227121) and [A corpus of nonspeech sounds](web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/HuCorpus.html) to train and test the net. Training, validation set and test set contained separate parts of the speech and noise files, such that all test files were previously unseen by the net. 

A few enhancement samples are provided on [Dropbox](https://www.dropbox.com/sh/gps8xzvya9cftp9/AAAp6f7eGHCmoC3MFqeSrXiYa?dl=0).



The work is inspired by the following repositories:
* [pix2pix](https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py)
* [segan](https://github.com/santi-pdp/segan)
* [se_relativisticgan](https://github.com/deepakbaby/se_relativisticgan)


The original implementation SEGAN uses latent noise along with the noisy speech as input to the generator. The latent noise was a part of the original GAN in order to find a mapping from random noise to wanted distribution. In the speech enhancement setting, the goal is _simply_ to find a mapping from noisy to clean speech - if the mapping is good, it is not important that it is stochastic. We have therefore provided an alternative implementation without latent noise. (During testing did the original implementation with latent noise perform slightly better.)

## Folder structure:
### src:
Contains the implementation. The training and testing is run from _main.py_. General options are also specified there. File paths are specified according to the local machine the program was developed on, and the cluster Idun, and must be changed in order to run the program from a new computer. In the files _generator.py_ and _discriminator.py_ the implementations of the generator and discriminator are located, following the specified options in main. _data_loader.py_ is used to load speech and noise files during training and testing. _tools.py_ contains other help functions.
### build:
Batch script used to run the code on NTNU's cluster Idun.
### logs:
Folder where validation and training error is saved during training.
### results:
The enhanced test set is stored here.
### results_test_sample:
The enhanced validation set is stored here.
### spectrograms:
Spectrograms of the noise files used to validate and test the GAN.



