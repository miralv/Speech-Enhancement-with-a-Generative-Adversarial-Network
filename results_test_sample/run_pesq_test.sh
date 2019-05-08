#!/bin/bash

# Run a bash script that calculates the pesq score for the files in the results folder

file_save="./file_save_09.05.19.out"

echo Calculate PESQ score: >> ${file_save}
echo file_name, score: >> ${file_save}
echo Enhanced files: >> ${file_save}

path_pesq="/home/shomec/m/miralv/Masteroppgave/P862/Software/source"

# start with the clean base, snr 10 for testing purposes
# right now is something wrong happening during the execution of this script
# the results get better when running directly in the terminal. why?

for clean_file in /home/shomec/m/miralv/Masteroppgave/Code/Deep-Learning-for-Speech-Separation/results_test_sample/*clean_*_snr_*.wav
do
  noisy_file=${clean_file}
  noisy_file=${clean_file/clean/noisy}

  echo -e ${noisy_file}: >> ${file_save}
  ${path_pesq}/PESQ +16000 ${clean_file} ${noisy_file} | grep 'Prediction : PESQ_MOS' >> ${file_save} 

  enhanced_file=${clean_file}
  enhanced_file=${clean_file/clean/enhanced}

  echo -e ${enhanced_file}: >> ${file_save}
  ${path_pesq}/PESQ +16000 ${clean_file} ${enhanced_file} | grep 'Prediction : PESQ_MOS' >> ${file_save} 



done


