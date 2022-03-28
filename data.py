import torch
torch.manual_seed(0)
import torchaudio
from functools import partial
from torch.utils.data import DataLoader

SAMPLE_RATE = 16000

def collect_audio_batch(batch, extra_noise=0., maxLen=600000):
    '''Collects a batch, should be list of tuples (audio_path <str>, list of int token <list>) 
       e.g. [(file1,txt1),(file2,txt2),...]
    '''
    def audio_reader(filepath):
        
        wav, sample_rate = torchaudio.load(filepath)
        if sample_rate != SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(wav)
        wav = wav.reshape(-1)
        if wav.shape[-1] >= maxLen:
            print(f'{filepath} has len {wav.shape}, truncate to {maxLen}')
            wav = wav[:maxLen]
            print(wav.shape)
        wav += extra_noise * torch.randn_like(wav)
        
        return wav

    # Bucketed batch should be [[(file1,txt1),(file2,txt2),...]]
    if type(batch[0]) is not tuple:
        batch = batch[0]

    # Read batch
    file, audio_feat, audio_len, text = [], [], [], []
    with torch.no_grad():
        for b in batch:
            feat = audio_reader(str(b[0])).numpy()
            file.append(str(b[0]).split('/')[-1].split('.')[0])
            audio_feat.append(feat)
            audio_len.append(len(feat))
            text.append(b[1])

    # Descending audio length within each batch
    audio_len, file, audio_feat, text = zip(*[(feat_len, f_name, feat, txt)
                                              for feat_len, f_name, feat, txt in sorted(zip(audio_len, file, audio_feat, text), reverse=True, key=lambda x:x[0])])

    return audio_len, audio_feat, text, file


def create_dataset(split, name, path, batch_size=1):
    ''' Interface for creating all kinds of dataset'''

    # Recognize corpus
    if name.lower() == "librispeech":
        from corpus.librispeech import LibriDataset as Dataset
    elif name.lower() == "chime":
        from corpus.CHiME import CHiMEDataset as Dataset
    elif name.lower() == "ted":
        from corpus.ted import TedDataset as Dataset
    elif name.lower() == "commonvoice":
        from corpus.commonvoice import CVDataset as Dataset
        
    else:
        raise NotImplementedError

    loader_bs = batch_size
    dataset = Dataset(split, batch_size, path)
    print(f'[INFO]    There are {len(dataset)} samples.')

    return dataset, loader_bs


def load_dataset(split=None, name='librispeech', path=None, batch_size=11, extra_noise=0., num_workers=4):
    ''' Prepare dataloader for training/validation'''
    dataset, loader_bs = create_dataset(split, name, path, batch_size)
    collate_fn = partial(collect_audio_batch, extra_noise=extra_noise)

    dataloader = DataLoader(dataset, batch_size=loader_bs, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers)
    return dataloader
