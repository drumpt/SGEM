import os
import copy
import random
import gc
import logging
import pickle
from datetime import datetime
from copy import deepcopy
import time
from collections import deque

import numpy as np
from sklearn.decomposition import PCA
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from info_nce import InfoNCE

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.lobes.augment import TimeDomainSpecAugment
from speechbrain.decoders.seq2seq import S2SRNNGreedySearcher
import speechbrain

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.common.parts.rnn import label_collate
from audio_augmentations import *
import sentencepiece

from jiwer import wer
import hydra
from omegaconf import OmegaConf, open_dict

from data import load_dataset


def forward_attn(args, model, greedy_searcher, wavs, gt_wavs=None):
    log_probs_lst = []

    enc_states = model.encode_batch(wavs, wav_lens=torch.ones(len(wavs)).to(args.device))
    enc_lens = torch.tensor([enc_states.shape[1]]).to(args.device)

    device = enc_states.device
    batch_size = enc_states.shape[0]
    memory = greedy_searcher.reset_mem(batch_size, device=device)

    inp_tokens = (enc_states.new_zeros(batch_size).fill_(greedy_searcher.bos_index).long())
    max_decode_steps = int(enc_states.shape[1] * greedy_searcher.max_decode_ratio)

    if gt_wavs == None:
        for _ in range(max_decode_steps):
            log_probs, memory, _ = greedy_searcher.forward_step(
                inp_tokens, memory, enc_states, enc_lens
            )
            log_probs_lst.append(log_probs)
            inp_tokens = log_probs.argmax(dim=-1)
    else:
        with torch.no_grad():
            gt_enc_states = model.encode_batch(gt_wavs, wav_lens=torch.ones(len(gt_wavs)).to(args.device))
            gt_enc_lens = torch.tensor([gt_enc_states.shape[1]]).to(args.device)

            gt_memory = greedy_searcher.reset_mem(batch_size, device=device)
            gt_inp_tokens = (gt_enc_states.new_zeros(batch_size).fill_(greedy_searcher.bos_index).long())
        for _ in range(max_decode_steps):
            log_probs, memory, _ = greedy_searcher.forward_step(
                gt_inp_tokens, memory, enc_states, enc_lens
            )

            with torch.no_grad():
                gt_log_probs, gt_memory, _ = greedy_searcher.forward_step(
                    gt_inp_tokens, gt_memory, gt_enc_states, gt_enc_lens
                )
                gt_inp_tokens = gt_log_probs.argmax(dim=-1)

            log_probs_lst.append(log_probs)
    return log_probs_lst


def forward_trans(args, model, wavs, lens, gt_wavs=None):
    log_probs_lst = []

    if gt_wavs == None:
        encoder_output, encoded_lengths = model(input_signal=wavs, input_signal_length=lens)
        encoder_output = encoder_output.transpose(1, 2)
        logitlen = encoded_lengths

        inseq = encoder_output  # [B, T, D]
        x, out_len, device = inseq, logitlen, inseq.device
        batchsize = x.shape[0]
        hypotheses = [rnnt_utils.Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batchsize)]
        hidden = None

        if model.decoding.decoding.preserve_alignments:
            for hyp in hypotheses:
                hyp.alignments = [[]]

        last_label = torch.full([batchsize, 1], fill_value=model.decoding.decoding._blank_index, dtype=torch.long, device=device)
        blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)

        max_out_len = out_len.max()
        for time_idx in range(max_out_len):
            f = x.narrow(dim=1, start=time_idx, length=1)  # [B, 1, D]

            not_blank = True
            symbols_added = 0

            blank_mask.mul_(False)
            blank_mask = time_idx >= out_len

            while not_blank and (model.decoding.decoding.max_symbols is None or symbols_added < model.decoding.decoding.max_symbols):
                if time_idx == 0 and symbols_added == 0 and hidden is None:
                    in_label = model.decoding.decoding._SOS
                else:
                    in_label = last_label
                if isinstance(in_label, torch.Tensor) and in_label.dtype != torch.long:
                    in_label = in_label.long()
                    g, hidden_prime = model.decoding.decoding.decoder.predict(None, hidden, False, batchsize)
                else:
                    if in_label == model.decoding.decoding._SOS:
                        g, hidden_prime = model.decoding.decoding.decoder.predict(None, hidden, False, batchsize)
                    else:
                        in_label = label_collate([[in_label.cpu()]])
                        g, hidden_prime = model.decoding.decoding.decoder.predict(in_label, hidden, False, batchsize)

                logp = model.decoding.decoding.joint.joint(f, g)
                if not logp.is_cuda:
                    logp = logp.log_softmax(dim=len(logp.shape) - 1)
                logp = logp[:, 0, 0, :]
                log_probs_lst.append(logp)

                if logp.dtype != torch.float32:
                    logp = logp.float()

                v, k = logp.max(1)
                del g

                k_is_blank = k == model.decoding.decoding._blank_index
                blank_mask.bitwise_or_(k_is_blank)
                del k_is_blank

                if model.decoding.decoding.preserve_alignments:
                    logp_vals = logp.to('cpu')
                    logp_ids = logp_vals.max(1)[1]
                    for batch_idx in range(batchsize):
                        if time_idx < out_len[batch_idx]:
                            hypotheses[batch_idx].alignments[-1].append(
                                (logp_vals[batch_idx], logp_ids[batch_idx])
                            )
                    del logp_vals

                if blank_mask.all():
                    not_blank = False
                    if model.decoding.decoding.preserve_alignments:
                        for batch_idx in range(batchsize):
                            if len(hypotheses[batch_idx].alignments[-1]) > 0:
                                hypotheses[batch_idx].alignments.append([])  # blank buffer for next timestep
                else:
                    blank_indices = (blank_mask == 1).nonzero(as_tuple=False)
                    if hidden is not None:
                        hidden_prime = model.decoding.decoding.decoder.batch_copy_states(hidden_prime, hidden, blank_indices)
                    elif len(blank_indices) > 0 and hidden is None:
                        hidden_prime = model.decoding.decoding.decoder.batch_copy_states(hidden_prime, None, blank_indices, value=0.0)
                    k[blank_indices] = last_label[blank_indices, 0]
                    last_label = k.clone().view(-1, 1)
                    hidden = hidden_prime
                    for kidx, ki in enumerate(k):
                        if blank_mask[kidx] == 0:
                            hypotheses[kidx].y_sequence.append(ki)
                            hypotheses[kidx].timestep.append(time_idx)
                            hypotheses[kidx].score += float(v[kidx])

                    symbols_added += 1
    else:
        encoder_output, encoded_lengths = model(input_signal=wavs, input_signal_length=lens)
        encoder_output = encoder_output.transpose(1, 2)
        logitlen = encoded_lengths

        # teacher-forcing
        gt_encoder_output, _ = model(input_signal=gt_wavs, input_signal_length=lens)
        gt_encoder_output = gt_encoder_output.transpose(1, 2)

        inseq = encoder_output  # [B, T, D]
        x, out_len, device = inseq, logitlen, inseq.device
        batchsize = x.shape[0]
        hypotheses = [rnnt_utils.Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batchsize)]
        hidden = None

        # teacher-forcing
        gt_x = gt_encoder_output
        gt_hypotheses = [rnnt_utils.Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batchsize)]
        gt_hidden = None

        if model.decoding.decoding.preserve_alignments:
            for hyp in hypotheses:
                hyp.alignments = [[]]

        last_label = torch.full([batchsize, 1], fill_value=model.decoding.decoding._blank_index, dtype=torch.long, device=device)
        blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)

        # teacher-forcing
        gt_last_label = torch.full([batchsize, 1], fill_value=model.decoding.decoding._blank_index, dtype=torch.long, device=device)
        gt_blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)

        batchsize = x.shape[0]

        max_out_len = out_len.max()
        for time_idx in range(max_out_len):
            f = x.narrow(dim=1, start=time_idx, length=1)  # [B, 1, D]

            not_blank = True
            symbols_added = 0

            blank_mask.mul_(False)
            blank_mask = time_idx >= out_len

            while not_blank and (model.decoding.decoding.max_symbols is None or symbols_added < model.decoding.decoding.max_symbols):
                if time_idx == 0 and symbols_added == 0 and hidden is None:
                    in_label = model.decoding.decoding._SOS
                else:
                    in_label = gt_last_label
                if isinstance(in_label, torch.Tensor) and in_label.dtype != torch.long:
                    in_label = in_label.long()
                    g, hidden_prime = model.decoding.decoding.decoder.predict(None, hidden, False, batchsize)
                else:
                    if in_label == model.decoding.decoding._SOS:
                        g, hidden_prime = model.decoding.decoding.decoder.predict(None, hidden, False, batchsize)
                    else:
                        in_label = label_collate([[in_label.cpu()]])
                        g, hidden_prime = model.decoding.decoding.decoder.predict(in_label, hidden, False, batchsize)

                logp = model.decoding.decoding.joint.joint(f, g)
                if not logp.is_cuda:
                    logp = logp.log_softmax(dim=len(logp.shape) - 1)
                logp = logp[:, 0, 0, :]
                log_probs_lst.append(logp)

                if logp.dtype != torch.float32:
                    logp = logp.float()

                v, k = logp.max(1)

                k_is_blank = k == model.decoding.decoding._blank_index

                blank_mask.bitwise_or_(k_is_blank)

                if model.decoding.decoding.preserve_alignments:
                    logp_vals = logp.to('cpu')
                    logp_ids = logp_vals.max(1)[1]
                    for batch_idx in range(batchsize):
                        if time_idx < out_len[batch_idx]:
                            hypotheses[batch_idx].alignments[-1].append(
                                (logp_vals[batch_idx], logp_ids[batch_idx])
                            )

                if blank_mask.all():
                    not_blank = False
                    if model.decoding.decoding.preserve_alignments:
                        for batch_idx in range(batchsize):
                            if len(hypotheses[batch_idx].alignments[-1]) > 0:
                                hypotheses[batch_idx].alignments.append([])  # blank buffer for next timestep
                else:
                    blank_indices = (blank_mask == 1).nonzero(as_tuple=False)
                    if hidden is not None:
                        hidden_prime = model.decoding.decoding.decoder.batch_copy_states(hidden_prime, hidden, blank_indices)
                    elif len(blank_indices) > 0 and hidden is None:
                        hidden_prime = model.decoding.decoding.decoder.batch_copy_states(hidden_prime, None, blank_indices, value=0.0)
                    k[blank_indices] = last_label[blank_indices, 0]
                    last_label = k.clone().view(-1, 1)
                    hidden = hidden_prime
                    for kidx, ki in enumerate(k):
                        if blank_mask[kidx] == 0:
                            hypotheses[kidx].y_sequence.append(ki)
                            hypotheses[kidx].timestep.append(time_idx)
                            hypotheses[kidx].score += float(v[kidx])
                    symbols_added += 1

            gt_f = gt_x.narrow(dim=1, start=time_idx, length=1)  # [B, 1, D]

            gt_not_blank = True
            gt_symbols_added = 0

            gt_blank_mask.mul_(False)
            gt_blank_mask = time_idx >= out_len

            while gt_not_blank and (model.decoding.decoding.max_symbols is None or gt_symbols_added < model.decoding.decoding.max_symbols):
                if time_idx == 0 and gt_symbols_added == 0 and gt_hidden is None:
                    gt_in_label = model.decoding.decoding._SOS
                else:
                    gt_in_label = gt_last_label
                if isinstance(gt_in_label, torch.Tensor) and gt_in_label.dtype != torch.long:
                    gt_in_label = gt_in_label.long()
                    gt_g, gt_hidden_prime = model.decoding.decoding.decoder.predict(None, gt_hidden, False, batchsize)
                else:
                    if gt_in_label == model.decoding.decoding._SOS:
                        gt_g, gt_hidden_prime = model.decoding.decoding.decoder.predict(None, gt_hidden, False, batchsize)
                    else:
                        gt_in_label = label_collate([[gt_in_label.cpu()]])
                        gt_g, gt_hidden_prime = model.decoding.decoding.decoder.predict(gt_in_label, gt_hidden, False, batchsize)

                gt_logp = model.decoding.decoding.joint.joint(gt_f, gt_g)
                if not gt_logp.is_cuda:
                    gt_logp = gt_logp.log_softmax(dim=len(gt_logp.shape) - 1)
                gt_logp = gt_logp[:, 0, 0, :]

                if gt_logp.dtype != torch.float32:
                    gt_logp = gt_logp.float()

                gt_v, gt_k = gt_logp.max(1)

                gt_k_is_blank = gt_k == model.decoding.decoding._blank_index

                gt_blank_mask.bitwise_or_(gt_k_is_blank)

                if model.decoding.decoding.preserve_alignments:
                    gt_logp_vals = gt_logp.to('cpu')
                    gt_logp_ids = gt_logp_vals.max(1)[1]
                    for batch_idx in range(batchsize):
                        if time_idx < out_len[batch_idx]:
                            gt_hypotheses[batch_idx].alignments[-1].append(
                                (gt_logp_vals[batch_idx], gt_logp_ids[batch_idx])
                            )

                if gt_blank_mask.all():
                    gt_not_blank = False
                    if model.decoding.decoding.preserve_alignments:
                        for batch_idx in range(batchsize):
                            if len(gt_hypotheses[batch_idx].alignments[-1]) > 0:
                                gt_hypotheses[batch_idx].alignments.append([])  # blank buffer for next timestep
                else:
                    blank_indices = (gt_blank_mask == 1).nonzero(as_tuple=False)
                    if gt_hidden is not None:
                        gt_hidden_prime = model.decoding.decoding.decoder.batch_copy_states(gt_hidden_prime, gt_hidden, blank_indices)
                    elif len(blank_indices) > 0 and gt_hidden is None:
                        gt_hidden_prime = model.decoding.decoding.decoder.batch_copy_states(gt_hidden_prime, None, blank_indices, value=0.0)
                    gt_k[blank_indices] = gt_last_label[blank_indices, 0]
                    gt_last_label = gt_k.clone().view(-1, 1)
                    gt_hidden = gt_hidden_prime
                    for kidx, ki in enumerate(gt_k):
                        if gt_blank_mask[kidx] == 0:
                            gt_hypotheses[kidx].y_sequence.append(ki)
                            gt_hypotheses[kidx].timestep.append(time_idx)
                            gt_hypotheses[kidx].score += float(gt_v[kidx])
                    gt_symbols_added += 1

    return log_probs_lst