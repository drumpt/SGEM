import os
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torchaudio
from IPython.display import Audio, display

from speechbrain.pretrained import EncoderDecoderASR
# from espnet2.bin.asr_inference import Speech2Text

from audio_augmentations import *
from speechbrain.lobes.augment import TimeDomainSpecAugment
from speechbrain.pretrained import SpectralMaskEnhancement
from asteroid.models import BaseModel

from jiwer import wer

from data import load_dataset
from main import collect_params

def get_logger(log_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"log_{time_string}.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

logger = get_logger(".")

split = ["test-other"]
dataset_name = "chime"
dataset_dir = "/home/server08/hdd0/changhun_workspace/CHiME3"
batch_size=1
extra_noise=0
steps = 10
lr = 2e-6

dataset = load_dataset(split, dataset_name, dataset_dir, batch_size, extra_noise)

metric_gan_plus = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus-voicebank",
)
conv_tasnet = BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k")

augmentation_list = [
    PolarityInversion(),
    Noise(min_snr=0.1, max_snr=0.5),
    Noise(min_snr=0.1, max_snr=1),
    Gain(),
    Reverb(sample_rate=16000),
    Delay(sample_rate=16000),
    HighLowPass(sample_rate=16000),
    PitchShift(n_samples=16000*5, sample_rate=16000, pitch_cents_min=-700, pitch_cents_max=700),
    PitchShift(n_samples=16000*5, sample_rate=16000, pitch_cents_min=-1400, pitch_cents_max=1400),
    TimeDomainSpecAugment(
            perturb_prob=1, drop_freq_prob=1, drop_chunk_prob=1, speeds=[80, 100, 120],
            drop_freq_count_low=10, drop_freq_count_high=20, drop_chunk_count_low=10, drop_chunk_count_high=20,
            drop_chunk_length_low=500, drop_chunk_length_high=1000, drop_chunk_noise_factor=0
    ),
    TimeDomainSpecAugment(
            perturb_prob=1, drop_freq_prob=1, drop_chunk_prob=1, speeds=[80, 100, 120],
            drop_freq_count_low=10, drop_freq_count_high=20, drop_chunk_count_low=10, drop_chunk_count_high=20,
            drop_chunk_length_low=500, drop_chunk_length_high=1000, drop_chunk_noise_factor=0
    ),
    # metric_gan_plus,
    # conv_tasnet
]

# augmentation test
model = EncoderDecoderASR.from_hparams("speechbrain/asr-crdnn-rnnlm-librispeech", run_opts={"device" : "cuda"})

gt_texts = []
transcriptions_list = [[] for _ in range(1 + len(augmentation_list))]

for batch_idx, batch in enumerate(dataset):
    if batch_idx >= 300:
        break
    lens, wavs, texts, files = batch
    wavs = torch.tensor(wavs)
    gt_texts += texts
    logger.info(f"{batch_idx}")
    logger.info(f"ground truth text: {texts}")

    model.eval()
    ori_transcription, _ = model.transcribe_batch(wavs, wav_lens=torch.FloatTensor([wavs.shape[1]]))
    transcriptions_list[0] += ori_transcription
    ori_wer = wer(list(texts), list(ori_transcription))
    logger.info(f"original text : {ori_transcription}")
    logger.info(f"original WER: {ori_wer}")

    for i, augmentation in enumerate(augmentation_list):
        if isinstance(augmentation, TimeDomainSpecAugment):
            aug_wavs = augmentation(wavs, lengths=torch.ones(1))
        elif isinstance(augmentation, SpectralMaskEnhancement):
            aug_wavs = augmentation.enhance_batch(wavs, lengths=torch.ones(1))
        # elif isinstance(augmentation, BaseModel):
        #     # print(f"wavs.shape : {wavs.shape}")
        #     aug_wavs = augmentation.forward_wav(wavs)
        #     # print(f"aug_wavs.shape : {aug_wavs.shape}")
        else:
            aug_wavs = augmentation(wavs)
        aug_transcription, _ = model.transcribe_batch(aug_wavs, wav_lens=torch.FloatTensor([wavs.shape[1]]))
        transcriptions_list[i + 1] += aug_transcription
        aug_wer = wer(list(texts), list(aug_transcription))
        logger.info(f"aug text : {aug_transcription}")
        logger.info(f"aug WER: {aug_wer}\n")
    logger.info("\n\n\n")

for transcriptions in transcriptions_list:
    logger.info(wer(list(gt_texts), list(transcriptions)))