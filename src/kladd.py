
# prepare a sample test

if TEST:
if options['Idun']:
    options['audio_path_test'] = "/home/miralv/Master/Audio/sennheiser_1/part_1/Test/group_12/p1_g12_m1_1_t-a0001.wav"
    options['noise_path_test'] = "/home/miralv/Master/Audio/Nonspeech/Test"
else:
    # options['audio_path_test'] = "/home/shomec/m/miralv/Masteroppgave/Code/sennheiser_1/part_1/Test/group_12/p1_g12_m1_1_t-a0001.wav"
    options['audio_folder_test'] = "/home/shomec/m/miralv/Masteroppgave/Code/sennheiser_1/part_1/Test/Selected"
    options['noise_folder_test'] = "/home/shomec/m/miralv/Masteroppgave/Code/Nonspeech_v2/Test" # /Train or /Validate or /Test

print("Test the model on unseen noises and voices.\n\n")
noise_list = glob.glob(options['noise_folder_test'] + "/*.wav")
speech_list = glob.glob(options['audio_folder_test'] + "/*-c*.wav") # Want only the unique sentences


# Holder med to speech files
# og et par typer noise

def run_sample_test(options,speech_list,noise_list,G):
    SNR_dBs = options['snr_dbs_test']
    for speech_path in speech_list:
        options['audio_path_test'] = speech_path
        for noise_path in noise_list:
            options['noise_path_test'] = noise_path
            clean, mixed, z = prepare_test(options) #(snr_dbs, nwindows, windowlength)
            for i,snr_db in enumerate(SNR_dBs):
                audios_mixed = np.expand_dims(mixed[i], axis=2)

                # Condition on B and generate a translated version
                G_out = G.predict([audios_mixed, z[i]]) 

                # Postprocess = upscale from [-1,1] to int16
                clean_res,_ = postprocess(clean[i,:,:], coeff = options['pre_emph'])
                mixed_res,_ = postprocess(mixed[i,:,:], coeff = options['pre_emph'])
                G_enhanced,_ = postprocess(G_out,coeff = options['pre_emph'])

                ## Save for listening
                if not os.path.exists("./results_test_sample"):
                    os.makedirs("./results_test_sample")

                # Want to save clean, enhanced and mixed. 
                sr = options['sample_rate']
    
                if noise_path[-7]=='n':
                    path_enhanced = "./results_test_sample/epoch_%d_enhanced_%s_%s_snr_%d.wav" % (epoch, speech_path[-16:-4],noise_path[-7:-4], snr_db)# sentence id, noise id, snr_db
                    path_noisy = "./results_test_sample/epoch_%d_noisy_%s_%s_snr_%d.wav" % (epoch, speech_path[-16:-4],noise_path[-7:-4], snr_db)
                    path_clean = "./results_test_sample/epoch_%d_clean_%s_%s_snr_%d.wav" % (epoch, speech_path[-16:-4],noise_path[-7:-4], snr_db)

                else:
                    path_enhanced = "./results_test_sample/epoch_%d_enhanced_%s_%s_snr_%d.wav" % (epoch, speech_path[-16:-4], noise_path[-16:-4], snr_db)
                    path_noisy = "./results_test_sample/epoch_%d_noisy_%s_%s_snr_%d.wav" % (epoch, speech_path[-16:-4], noise_path[-16:-4], snr_db)
                    path_clean = "./results_test_sample/epoch_%d_clean_%s_%s_snr_%d.wav" % (epoch, speech_path[-16:-4], noise_path[-16:-4], snr_db)

                # Because pesq is testing corresponding clean, noisy and enhanced, must clean be stored similarly
                saveAudio(clean_res, path_clean, sr) 
                saveAudio(mixed_res, path_noisy, sr)
                saveAudio(G_enhanced, path_enhanced, sr)