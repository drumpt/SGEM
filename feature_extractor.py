import pickle
from IPython.display import Audio, display

from sklearn.decomposition import PCA, TruncatedSVD
import torch
import torch.nn as nn
import torchaudio

from speechbrain.pretrained import EncoderDecoderASR
# from espnet2.bin.asr_inference import Speech2Text

from audio_augmentations import *
from speechbrain.lobes.augment import TimeDomainSpecAugment

from jiwer import wer

from data import load_dataset

def collect_params(model, train_params, bias_only=False):
    params = []
    names = []

    for np, p in model.named_parameters():
        collect = False
        if "all" in train_params:
            collect = True
        if 'enc' in train_params and 'enc' in str(np):
            collect = True
        if 'dec' in train_params and 'dec' in str(np):
            collect = True
        if 'linear' in train_params and 'fc' in str(np):
            collect = True
        if 'LN' in train_params and 'norm' in str(np):
            collect = True

        if collect:
            p.requires_grad = True
            params.append(p)
            names.append(str(np))

    return params, names


train_split = ["train-clean-100"]
# dataset_name = "chime"
# dataset_dir = "/home/server08/hdd0/changhun_workspace/CHiME3"
dataset_name = 'librispeech'
dataset_dir = '/home/server17/hdd/changhun_workspace/LibriSpeech'
source_dir = "source_full.pkl"
subspace_dir = "subspace_full.pkl"

batch_size=1
extra_noise=0.00
steps = 10
lr = 2e-6
n_components = 128

train_dataset = load_dataset(train_split, dataset_name, dataset_dir, batch_size, extra_noise)

model = EncoderDecoderASR.from_hparams("speechbrain/asr-crdnn-rnnlm-librispeech", run_opts={"device" : "cuda"})

feature_list = []

for batch_idx, batch in enumerate(train_dataset):
    lens, wavs, texts, files = batch
    wavs = torch.tensor(wavs)

    feature_batch = model.encode_batch(wavs,wav_lens=torch.FloatTensor([wavs.shape[1]]))
    for feature in feature_batch:
        feature_list.extend(list(feature.detach().cpu()))

feature_tensor = torch.stack(feature_list, dim=0)
# with open(source_dir, "wb") as f:
#     pickle.dump(feature_tensor, f)

pca = PCA(n_components=n_components).fit(feature_tensor)
with open(subspace_dir, "wb") as f:
    pickle.dump(torch.tensor(pca.components_), f)

phi = nn.Linear(in_features=512, out_features=128, bias=False)
phi.weight.data = torch.tensor(pca.components_).clone()