import torch
torch.manual_seed(0)
import torchaudio
from functools import partial
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence


def collect_audio_batch(batch, split, half_batch_size_wav_len=300000):
    '''Collects a batch, should be list of tuples (audio_path <str>, list of int token <list>) 
       e.g. [(file1,txt1),(file2,txt2),...]
    '''
    def audio_reader(filepath):
        wav, sample_rate = torchaudio.load(filepath)
        wav = wav.reshape(-1)
        # wav += 0.01 * torch.randn_like(wav)
        return wav

    # Bucketed batch should be [[(file1,txt1),(file2,txt2),...]]
    if type(batch[0]) is not tuple:
        batch = batch[0]

    # Make sure that batch size is reasonable
    first_len = audio_reader(str(batch[0][0])).size(0)
    if split == 'train':
        if first_len > half_batch_size_wav_len and len(batch) > 1:
            batch = batch[:len(batch)//2]

    # Read batch
    file, audio_feat, audio_len, text = [], [], [], []
    with torch.no_grad():
        for b in batch:
            file.append(str(b[0]).split('/')[-1].split('.')[0])
            feat = audio_reader(str(b[0])).numpy()
            audio_feat.append(feat)
            audio_len.append(len(feat))
            text.append(b[1])

    # Descending audio length within each batch
    audio_len, file, audio_feat, text = zip(*[(feat_len, f_name, feat, txt)
                                              for feat_len, f_name, feat, txt in sorted(zip(audio_len, file, audio_feat, text), reverse=True, key=lambda x:x[0])])

    return audio_len, audio_feat, text, file


def create_dataset(split, name, path, batch_size=12):
    ''' Interface for creating all kinds of dataset'''

    # Recognize corpus
    if name.lower() == "librispeech":
        from librispeech import LibriDataset as Dataset
    # elif name.lower() == "snips":
    #     from .corpus.snips import SnipsDataset as Dataset
    # elif name.lower() == 'libriphone':
    #     from .corpus.libriphone import LibriPhoneDataset as Dataset
    else:
        raise NotImplementedError

    loader_bs = batch_size
    dataset = Dataset(split, batch_size, path)

    return dataset, loader_bs


def load_dataset(split=None, name='librispeech', path=None, batch_size=12, num_workers=4):
    ''' Prepare dataloader for training/validation'''
    dataset, loader_bs = create_dataset(split, name, path, batch_size)
    collate_fn = partial(collect_audio_batch, split=split)

    dataloader = DataLoader(dataset, batch_size=loader_bs, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers)
    return dataloader

if __name__ == '__main__':
    dataset = load_dataset('test-other', 'librispeech', '/home/daniel094144/Daniel/data/LibriSpeech', 12)