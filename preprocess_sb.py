import soundfile as sf
import librosa
import os
import numpy as np
import re 

stm_path = '/home/daniel094144/SpeechTent/data/eval2000/stm'
audio_path  = '/home/daniel094144/data/Switchboard/eval2000_wav'
save_audio_dir = '/home/daniel094144/data/Switchboard/eval2000_wav_segment'
save_text_dir = '/home/daniel094144/data/Switchboard/eval2000_transcription'

SAMPLE_RATE = 16000

# preprocess text 
def preprocess_text(text):
    text = text.replace("(%HESITATION)", "")
    text = text.replace("-", " ")
    text = re.sub("[^ A-Z']", "", text)
    text = ' '.join(text.split())
    
    return text

with open(stm_path, 'r') as f: 
    curr＿file = None
    for line in f: 
        l = line.split()
        name = l[0]
        s = float(l[3])
        e = float(l[4])
        txt = ' '.join(l[6:])
        if curr_file != name: 
            print('---new---')
            print(name)
            wav, sr = sf.read(os.path.join(audio_path, name+'.wav'))
            wav = wav.mean(-1)
            
        start_idx = int(s * sr)
        end_idx = int(e * sr)
        segment = wav[start_idx: end_idx]
        # resample to 16k hz
        segment = librosa.resample(segment, orig_sr=sr, target_sr=SAMPLE_RATE)

        norm_txt = preprocess_text(txt)    
        apath =  os.path.join(save_audio_dir, '-'.join([name, l[3], l[4]])+'.wav')
        tpath =  os.path.join(save_text_dir, '-'.join([name, l[3], l[4]])+'.txt')
        
        
        sf.write(apath, segment, SAMPLE_RATE)
        with open(tpath, 'w') as tf:
            tf.write(norm_txt)
        curr_file = name



