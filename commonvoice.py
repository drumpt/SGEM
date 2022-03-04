import re
from builtins import str as unicode
import pandas as pd 
import os 
from unicodedata import name
from tqdm import tqdm
from pathlib import Path
import os
from joblib import Parallel, delayed
from torch.utils.data import Dataset            

def preprocess_text(text):
    text = unicode(text)
    text = text.replace("i.e.", "that is")
    text = text.replace("e.g.", "for example")
    text = text.replace("Mr.", "Mister")
    text = text.replace("Mrs.", "Mistress")
    text = text.replace("Dr.", "Doctor")
    text = text.replace("-", " ")
    text = text.upper()
    text = re.sub("[^ A-Z']", "", text)
    text = ' '.join(text.split())
    
    return text

class CVDataset(Dataset):
    def __init__(self, split, bucket_size, path="/home/daniel094144/data/cv-corpus-5.1-2020-06-22/en", enhance=False, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        
        split = ['']
        apath = path + "/clips"
        tpath = path + "/test.tsv"

        df = pd.read_csv(tpath, sep='\t')
        text = df['sentence'].apply(preprocess_text).values
        file_list = df['path'].values
        file_list = [os.path.join(apath, f) for f in file_list]

        print(len(file_list), len(text))
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
