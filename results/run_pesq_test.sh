#!/bin/bash

# Run a bash script that calculates the pesq score for the files in the results folder

file_save="./file_save_06.05.19.out"

echo Calculate PESQ score: >> ${file_save}
echo file_name, score: >> ${file_save}
echo Enhanced files: >> ${file_save}

path_pesq="/home/shomec/m/miralv/Masteroppgave/P862/Software/source"
#/home/shomec/m/miralv/Masteroppgave/Code/Deep-Learning-for-Speech-Separation/results/

# start with the clean base
for clean_file in /home/shomec/m/miralv/Masteroppgave/Code/Deep-Learning-for-Speech-Separation/results/clean_*_snr_*.wav
do
  noisy_file=${clean_file}
  noisy_file=${clean_file/clean/noisy}

  echo -e ${noisy_file}: >> ${file_save}
  ${path_pesq}/PESQ +16000 ${clean_file} ${noisy_file} | grep 'Prediction : PESQ_MOS' >> ${file_save} 


  enhanced_file=${clean_file}
  enhanced_file=${clean_file/clean/enhanced}

  echo -e ${noisy_file}: >> ${file_save}
  ${path_pesq}/PESQ +16000 ${clean_file} ${enhanced_file} | grep 'Prediction : PESQ_MOS' >> ${file_save} 



done


