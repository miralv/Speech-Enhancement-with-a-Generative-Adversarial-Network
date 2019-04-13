#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --time=01:00:00 
#SBATCH --partition=EPICALL 
#SBATCH --job-name="GAN for SE"
#SBATCH --output=gan_first_try.out


HOMEDIR=${SLURM_SUBMIT_DIR}
cd ${HOMEDIR}
module load goolfc/2017b Python/3.6.3 cuDNN/7
source venv/bin/activate
nice python -um dopamine.discrete_domains.train --base_dir=//lustre1/work/ramunter/bdqn_cartpole --gin_files='dopamine/agents/bdqn/configs/dqn_cartpole.gin'