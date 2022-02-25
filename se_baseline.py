import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement

enhance_model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus-voicebank",
)

import glob
import os 

data_dirs = [
            '/home/daniel094144/data/TEDLIUM_release2/test/wav_segment', 
            '/home/daniel094144/data/Switchboard/eval2000_wav_segment', 
            "/home/daniel094144/data/CHiME3/data/audio/16kHz/enhanced/et05_bus_real",
            "/home/daniel094144/data/CHiME3/data/audio/16kHz/enhanced/et05_bus_simu",
            "/home/daniel094144/data/CHiME3/data/audio/16kHz/enhanced/et05_caf_real",
            "/home/daniel094144/data/CHiME3/data/audio/16kHz/enhanced/et05_caf_simu",
            "/home/daniel094144/data/CHiME3/data/audio/16kHz/enhanced/et05_ped_simu",
            "/home/daniel094144/data/CHiME3/data/audio/16kHz/enhanced/et05_str_real",
            "/home/daniel094144/data/CHiME3/data/audio/16kHz/enhanced/et05_str_simu",
            ]

for data_dir in data_dirs:
    output_dir = os.path.join(data_dir, 'se_wav') 
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for file in glob.glob(os.path.join(data_dir, "*.wav")):
        # Load and add fake batch dimension
        noisy = enhance_model.load_audio(file).unsqueeze(0)

        # # Add relative length tensor
        enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))

        # # Saving enhanced signal on disk
        fname = file.split("/")[-1]
        print(fname)
        torchaudio.save(os.path.join(output_dir, fname), enhanced.cpu(), 16000)