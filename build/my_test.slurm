#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --time=00:15:00 
#SBATCH --partition=EPICALL 
#SBATCH --job-name="GAN for SE"
#SBATCH --output=gan_first_try.out


HOMEDIR=/home/miralv/Master/Deep-Learning-for-Speech-Separation/src
cd ${HOMEDIR}
echo ${HOMEDIR}

module purge
module load GCC/6.4.0-2.28  CUDA/9.0.176  OpenMPI/2.1.1
module load goolfc/2017b
module load TensorFlow/1.7.0-Python-3.6.3
###module load cuDNN/7

source /home/miralv/Master/v_env/master_virtualenv/bin/activate
pip freeze
python --version

nice python main.py
