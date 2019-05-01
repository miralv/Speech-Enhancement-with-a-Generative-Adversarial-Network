#!/bin/bash

# Run a bash script that calculates the pesq score for the files in the results folder

file_save="./file_save_01.05.19.out"

echo Calculate PESQ score: >> ${file_save}
echo file_name, score: >> ${file_save}
clean_file="./clean.wav"
path_pesq="/home/shomec/m/miralv/Masteroppgave/P862/Software/source"
#/home/shomec/m/miralv/Masteroppgave/Code/Deep-Learning-for-Speech-Separation/results/

for noisy_file in /home/shomec/m/miralv/Masteroppgave/Code/Deep-Learning-for-Speech-Separation/results/enhanced_*.wav
do
  echo -e ${noisy_file}: >> ${file_save}
  ${path_pesq}/PESQ +16000 ${clean_file} ${noisy_file} | grep 'Prediction : PESQ_MOS' >> ${file_save} 
done




printf "\n\nNoisy files:\n" >> ${file_save}


for noisy_file in /home/shomec/m/miralv/Masteroppgave/Code/Deep-Learning-for-Speech-Separation/results/noisy_*.wav
do
  echo -e ${noisy_file}: >> ${file_save}
  ${path_pesq}/PESQ +16000 ${clean_file} ${noisy_file} | grep 'Prediction : PESQ_MOS' >> ${file_save} 
done

