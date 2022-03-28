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
    txt_list = os.path.join(tpath, "".join("/".join(file.split('/')[-2:]).split(".")[:-1])+'.trn')

    with open(txt_list, 'r') as fp:
        for line in fp:
            return ' '.join(line.split(' ')[1:]).strip('\n')
            


class CHiMEDataset(Dataset):
    def __init__(self, split, bucket_size, path="/home/daniel094144/data/CHiME3", enhance=False, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        
        split = ['et05_bus_real', 'et05_bus_simu', 'et05_caf_real', 'et05_caf_simu', 'et05_ped_simu', 'et05_str_real', 'et05_str_simu']
        apath = path + "/data/audio/16kHz/enhanced"
        tpath = path + "/data/transcriptions"

        file_list = []
        for s in split: 
            split_list = list(Path(os.path.join(apath, s)).glob("*.wav"))
            file_list += split_list
        
        text = []
        for f in tqdm(file_list, desc='Read text'):
            transcription = read_text(tpath, str(f))
            text.append(transcription)

        if enhance: 
            file_list = []
            for s in split: 
                split_list = list(Path(os.path.join(os.path.join(apath, s), 'se_wav')).glob("*.wav"))
                file_list += split_list

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
