import re
from tqdm import tqdm
from pathlib import Path
import os
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from builtins import str as unicode
from unicodedata import name


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


class ValDataset(Dataset):
    def __init__(self, bucket_size, path, enhance=False, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        
        apath = os.path.join(path, "noisy_testset_wav")
        tpath = os.path.join(path, "testset_txt")

        file_list, text_list = [], []
        for wav in sorted(os.listdir(apath)):
            if not wav.endswith(".wav"):
                continue
            file_list.append(os.path.join(apath, wav))
        for txt_file in sorted(os.listdir(tpath)):
            if not txt_file.endswith(".txt"):
                continue
            with open(os.path.join(tpath, txt_file), "r") as f:
                txt = f.read()
                txt = preprocess_text(txt)
            text_list.append(txt)

        assert len(file_list) == len(text_list)

        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(file_list, text_list), reverse=not ascending, key=lambda x:len(x[1]))])

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