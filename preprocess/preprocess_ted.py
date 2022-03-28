import soundfile as sf
import os
import re 

stm_path = '/home/daniel094144/data/TEDLIUM_release2/test/stm'
audio_path  = '/home/daniel094144/data/TEDLIUM_release2/test/wav'
save_audio_dir = '/home/daniel094144/data/TEDLIUM_release2/test/wav_segment'
save_text_dir = '/home/daniel094144/data/TEDLIUM_release2/test/transcription'

SAMPLE_RATE = 16000

# preprocess text 
def preprocess_text(text):
    text = text.upper()
    text = text.replace(" '", "'")
    text = text.replace("-", " ")
    text = re.sub("[^ A-Z']", "", text)
    text = ' '.join(text.split())
    
    return text

skip = 'inter_segment_gap'

import glob

for stm_path in glob.glob(os.path.join(stm_path, '*.stm')):
    with open(stm_path, 'r') as f: 
        curr＿file = None
        for line in f: 
            l = line.split()
            name = l[0]
            if l[2] == skip: 
                continue
            s = float(l[3])
            e = float(l[4])
            txt = ' '.join(l[6:])
            if curr_file != name: 
                print('---new---')
                print(name)
                wav, sr = sf.read(os.path.join(audio_path, name+'.wav'))
                print(wav.shape)
                
            start_idx = int(s * sr)
            end_idx = int(e * sr)
            segment = wav[start_idx: end_idx]

            norm_txt = preprocess_text(txt)    
            apath =  os.path.join(save_audio_dir, '-'.join([name, l[3], l[4]])+'.wav')
            tpath =  os.path.join(save_text_dir, '-'.join([name, l[3], l[4]])+'.txt')
            
            
            sf.write(apath, segment, SAMPLE_RATE)
            with open(tpath, 'w') as tf:
                tf.write(norm_txt)
            curr_file = name



