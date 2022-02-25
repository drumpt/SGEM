from unicodedata import name
from tqdm import tqdm
from pathlib import Path
import os
from joblib import Parallel, delayed
from torch.utils.data import Dataset


def read_text(tpath, file):
    '''Get transcription of target wave file, 
       it's somewhat redundant for accessing each txt multiplt times,
       but it works fine with multi-thread'''
    file = file.split('/')[-1].replace('wav', 'txt')
    txt_list = os.path.join(tpath, file)

    with open(txt_list, 'r') as fp:
        for line in fp:
            return line.strip('\n')
            


class TedDataset(Dataset):
    def __init__(self, split, bucket_size, path="/home/daniel094144/data/TEDLIUM_release2/test", enhance=False, ascending=True):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        
        split = ['']
        apath = path + "/wav_segment"
        tpath = path + "/transcription"

        file_list = []
        for s in split: 
            if enhance: 
                split_list = list(Path(os.path.join(os.path.join(apath, 'se_wav'), s)).glob("*.wav"))
            else:  
                split_list = list(Path(os.path.join(apath, s)).glob("*.wav"))
            file_list += split_list
        
        text = []
        filtered_file_list = []
        for f in tqdm(file_list, desc='Read text'):
            transcription = read_text(tpath, str(f))
            # print(transcription)
            if transcription == None: 
                pass
            # elif len(transcription.split()) <= 3:
            #     pass
            else:
                filtered_file_list.append(f)
                text.append(transcription)

        print(len(filtered_file_list), len(text))
        file_list = filtered_file_list
        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(file_list, text), reverse=not ascending, key=lambda x:len(x[1]))])

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list)-self.bucket_size, index)
            return [(f_path, txt) for f_path, txt in
                    zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
        else:
            return self.file_list[index], self.text[index]

    def __len__(self):
        return len(self.file_list)
