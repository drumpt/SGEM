import os
import random
import gc
import logging
import pickle
import shelve
from copy import deepcopy
import time
from datetime import datetime
from collections import deque

import hydra
from omegaconf import OmegaConf
import numpy as np
from sklearn.decomposition import PCA
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Beta
from info_nce import InfoNCE
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2ProcessorWithLM
import speechbrain
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.lobes.augment import TimeDomainSpecAugment
from speechbrain.decoders.seq2seq import S2SRNNGreedySearcher, S2SBaseSearcher
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.parts.submodules import rnnt_greedy_decoding
from nemo.collections.asr.parts.submodules.rnnt_beam_decoding import BeamRNNTInfer
from pyctcdecode.constants import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_HOTWORD_WEIGHT,
    DEFAULT_MIN_TOKEN_LOGP,
    DEFAULT_PRUNE_LOGP,
    DEFAULT_PRUNE_BEAMS
)
from audio_augmentations import *
from jiwer import wer

from data import load_dataset
from forward import *
from loss import *



def get_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_handler = logging.FileHandler(os.path.join(args.log_dir, f"log_{time_string}.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_model(args, original):
    if args.asr == "facebook/wav2vec2-base-960h":
        model = Wav2Vec2ForCTC.from_pretrained(args.asr).requires_grad_(True).eval()
        if 'cuda' in args.device:
            model = model.cuda()
    elif args.asr == "speechbrain/asr-crdnn-rnnlm-librispeech":
        model = EncoderDecoderASR.from_hparams(args.asr, run_opts={"device": args.device}).requires_grad_(True).eval()
    elif args.asr == "speechbrain/asr-crdnn-transformerlm-librispeech":
        model = EncoderDecoderASR.from_hparams(args.asr, run_opts={"device": args.device}).requires_grad_(True).eval()
    elif args.asr == "pretrained_models/stt_en_conformer_transducer_small.nemo":
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(args.asr).to(args.device).requires_grad_(True).eval()
    elif args.asr == "pretrained_models/stt_en_conformer_transducer_large.nemo":
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(args.asr).to(args.device).requires_grad_(True).eval()
    if original:
        model = configure_model(model)
    return model


def collect_params_ctc(model, train_params, bias_only=False):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    trainable = []
    if bias_only:
        trainable = ['bias']
    else: 
        trainable = ['weight', 'bias']

    for nm, m in model.named_modules():
        if "all" in train_params:
            for np, p in m.named_parameters():
                p.requires_grad = True
                if not f"{nm}.{np}" in names:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        if "feature" in train_params:
            if len(str(nm).split('.')) > 1:
                if str(nm).split('.')[1] == 'feature_extractor' or str(nm).split('.')[1] == 'feature_projection':
                    for np, p in m.named_parameters():
                        p.requires_grad = True
                        if not f"{nm}.{np}" in names:
                            params.append(p)
                            names.append(f"{nm}.{np}")
        if "LN" in train_params:
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in trainable:  
                        p.requires_grad = True
                        if not f"{nm}.{np}" in names:
                            params.append(p)
                            names.append(f"{nm}.{np}")
    return params, names


def collect_params_attn(model, train_params, bias_only=False):
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


def collect_params_trans(model, train_params, bias_only=False):
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
        if 'joint' in train_params and 'joint' in str(np):
            collect = True
        if 'linear' in train_params and 'joint_net' in str(np):
            collect = True
        if 'LN' in train_params and 'norm' in str(np):
            collect = True

        if collect:
            p.requires_grad = True
            params.append(p)
            names.append(str(np))
    return params, names


def configure_model(model):
    """Configure model for use with tent."""
    model.requires_grad_(False)
    return model


def freeze_norm_stats(model):
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            m.track_running_stats = False


def eval_except_for_rnn(model):
    model.eval()
    if isinstance(model, EncoderDecoderASR):
        for nm, m in model.named_modules():
            if 'rnn' in nm.lower() or 'lstm' in nm.lower():
                m.train()
                m.dropout = 0
    # elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
    #     for nm, m in model.named_modules():
    #         if 'rnn' in nm.lower() or 'lstm' in nm.lower():
    #             m.train()
    #             if hasattr(m, 'dropout') and isinstance(m.dropout, float):
    #                 m.dropout = 0


def get_optimizer(params, opt_name='AdamW', lr=1e-4, beta=0.9, weight_decay=0., scheduler=None, step_size=1, gamma=0.7):
    opt = getattr(torch.optim, opt_name)
    if opt_name == 'Adam':       
        optimizer = opt(params, lr=lr, betas=(beta, 0.999), weight_decay=weight_decay)
    else: 
        optimizer = opt(params, lr=lr, weight_decay=weight_decay)
    
    if scheduler is not None: 
        return optimizer, eval(scheduler)(optimizer, step_size=step_size, gamma=gamma)
    return optimizer, None


def get_augmentation(args):
    weak_augmentation_list = [
        Noise(min_snr=0.01, max_snr=0.05),
        Gain(),
        Reverb(sample_rate=16000),
        HighLowPass(sample_rate=16000),
    ]
    strong_augmentation_list = [
        Noise(min_snr=0.1, max_snr=0.5),
        PitchShift(n_samples=16000, sample_rate=16000, pitch_cents_min=-700, pitch_cents_max=700),
        TimeDomainSpecAugment(
            perturb_prob=0, drop_freq_prob=1, drop_chunk_prob=1, speeds=[95, 100, 105],
            drop_freq_count_low=3, drop_freq_count_high=5, drop_chunk_count_low=3, drop_chunk_count_high=5,
            drop_chunk_length_low=500, drop_chunk_length_high=1000
        ),
    ]
    return weak_augmentation_list, strong_augmentation_list


def apply_augmentation(args, augmentation_list, wavs):
    if args.aug_method == "augmix":
        return apply_augmix(args, augmentation_list, wavs)
    augmentation = np.random.choice(augmentation_list)
    if isinstance(augmentation, TimeDomainSpecAugment):
        aug_wavs = augmentation(wavs, torch.ones(len(wavs)).to(wavs.device))
    else:
        aug_wavs = augmentation(wavs.cpu())
    return aug_wavs


def apply_augmix(args, augmentation_list, wavs, k=3, alpha=1.0):
    wavs_augmix = torch.zeros_like(wavs).to(wavs.device)
    w_list = torch.distributions.dirichlet.Dirichlet(torch.tensor([alpha] * k)).sample()
    for i in range(k):
        num_augs = torch.randint(low=1, high=4, size=(1,))[0]
        wavs_aug = wavs.clone().cpu()
        for _ in range(num_augs):
            augmentation = np.random.choice(augmentation_list)
            if isinstance(augmentation, TimeDomainSpecAugment):
                wavs_aug = augmentation(wavs_aug, torch.ones(len(wavs)))
            else:
                wavs_aug = augmentation(wavs_aug.cpu())
        wavs_augmix += w_list[i] * wavs_aug.to(wavs.device)
    m = torch.distributions.beta.Beta(alpha, alpha).sample()
    return m * wavs + (1 - m) * wavs_augmix


def copy_model_and_optimizer(model, optimizer, scheduler):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    if scheduler is not None:
        scheduler_state = deepcopy(scheduler.state_dict())
        return model_state, optimizer_state, scheduler_state
    else:
        return model_state, optimizer_state, None


def load_model_and_optimizer(model, optimizer, scheduler, model_state, optimizer_state, scheduler_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    model.eval()
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
        return model, optimizer, scheduler
    else: 
        return model, optimizer, None


def generate_adversarial_example(wavs, model, target_snr=15, lr=1e-4, n_steps=5):
    import math
    adv_wavs = []
    l1_loss = nn.L1Loss()
    for wav in wavs:
        wav = wav.unsqueeze(0).requires_grad_(False)
        noise = torch.zeros_like(wav).requires_grad_(True)
        for _ in range(n_steps):
            adv_wav = wav + noise
            clean_encoded = model.encode_batch(wav, wav_lens=torch.ones(1))
            adv_encoded = model.encode_batch(adv_wav, wav_lens=torch.ones(1))

            noise.grad = None
            model.zero_grad()

            loss = l1_loss(clean_encoded, adv_encoded)
            loss.backward()

            noise = noise - lr * noise.grad.sign()
            epsilon = torch.norm(wav) / (math.sqrt(wav.shape[0] * wav.shape[1])) * (10 ** (- target_snr / 10))
            noise = torch.clamp(noise, - epsilon, epsilon).detach()
            noise.requires_grad_(True)
        adv_wav = wav + noise
        adv_wavs.append(adv_wav.squeeze(0))
    return torch.stack(adv_wavs, dim=0)


def get_instance_from_queue(args, method, wavs, probs):
    out_wavs, out_lens, out_hash_values = [], [], []

    if args.n_neighbors <= 0:
        selected_instances = []
    elif len(memory_queue) <= args.n_neighbors:
        selected_instances = list(memory_queue)
    elif method == "random":
        selected_instances = random.sample(list(memory_queue), args.n_neighbors)
    elif method == "latest":
        selected_instances = list(memory_queue)[-args.n_neighbors:]
    elif method == "informative":
        new_mean_probs = torch.mean(probs.view(-1, probs.shape[-1]), dim=0)
        previous_instances = list(memory_queue)
        selected_instances = []
        for wav, prob, hash_value in previous_instances:
            entropy = torch.mean(- torch.sum(prob * torch.log(prob), dim=-1)) # mean over tokens
            mean_probs = torch.mean(prob, dim=0).to(wavs.device)
            js_div = js_divergence(new_mean_probs, mean_probs)
            selected_instances.append((wav, entropy / js_div, hash_value))
        selected_instances = sorted(selected_instances, key=lambda x: x[1], reverse=True)[:args.n_neighbors]
    elif method == "similar":
        new_mean_probs = torch.mean(probs.view(-1, probs.shape[-1]), dim=0)
        previous_instances = list(memory_queue)
        selected_instances = []
        for wav, prob, hash_value in previous_instances:
            mean_probs = torch.mean(prob, dim=0).to(wavs.device)
            js_div = js_divergence(new_mean_probs, mean_probs)
            selected_instances.append((wav, 1 / js_div, hash_value))
        selected_instances = sorted(selected_instances, key=lambda x: x[1], reverse=True)[:args.n_neighbors]

    for wav, _, hash_value in selected_instances:
        out_wavs.append(wav.to(wavs.device))
        out_lens.append(torch.tensor(len(wav)).to(wavs.device))
        out_hash_values.append(hash_value)

    if len(out_wavs) > 0:
        out_wavs = pad_sequence(out_wavs, batch_first=True)
    return out_wavs, out_lens, out_hash_values


def forward_and_adapt_ctc(args, model, teacher_model, processor, optimizer, scheduler, wavs, lens):
    optimizer.zero_grad()

    if "original" in args.method or "em_uncertainty" in args.method or "em_sparse" in args.method or "memo_dropout" in args.method:
        for i, wav in enumerate(wavs):

            import time
            current = time.time()

            wav = wav[:lens[i]].unsqueeze(0)
            outputs = model(wav).logits
            
            predicted_ids = torch.argmax(outputs, dim=-1)
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()

            print(f"time.time() - current: {time.time() - current}")

            if args.em_coef > 0:
                if "original" in args.method:
                    if args.not_blank:
                        e_loss = softmax_entropy(outputs / args.temp)[non_blank].mean(0).mean()
                    else: 
                        e_loss = softmax_entropy(outputs / args.temp).mean(0).mean() 
                elif "em_uncertainty" in args.method:
                    if args.not_blank:
                        frame_weight = F.normalize(softmax_entropy(outputs)[non_blank], p=1, dim=-1).detach()
                        e_loss = torch.sum(frame_weight * softmax_entropy(outputs / args.temp)[non_blank], dim=-1).mean()
                    else:
                        frame_weight = F.normalize(softmax_entropy(outputs), dim=-1).detach()
                        e_loss = torch.sum(frame_weight * softmax_entropy(outputs / args.temp), dim=-1).mean()
                elif "em_sparse" in args.method:
                    if args.not_blank:
                        selected_frame = non_blank & torch.where(softmax_entropy(outputs, dim=-1) < args.entropy_threshold, 1, 0).bool()
                        e_loss = softmax_entropy(outputs / args.temp)[selected_frame].mean(0).mean()
                    else:
                        selected_frame = torch.where(softmax_entropy(outputs, dim=-1) < args.entropy_threshold, 1, 0).bool()
                        e_loss = softmax_entropy(outputs / args.temp)[selected_frame].mean(0).mean()
                elif "memo_dropout" in args.method:
                    e_loss = 0
                    model.train()
                    for _ in range(5):
                        outputs = model(wav).logits
                        e_loss += (softmax_entropy(outputs / args.temp)[non_blank].mean(0).mean()) / 5
                    model.eval()

                (args.em_coef * e_loss / (len(wavs))).backward(retain_graph=True)

            if 1 - args.em_coef > 0:
                c_loss = mcc_loss(outputs / args.temp, args.reweight)
                ((1 - args.em_coef) * c_loss / (len(wavs))).backward(retain_graph=True)
    if "cr" in args.method:
        weak_augmentation_list, strong_augmentation_list = get_augmentation(args)

        ce_loss = nn.CrossEntropyLoss()
        for i, sub_wav in enumerate(wavs): # element-wise iteration
            sub_wav = sub_wav.unsqueeze(0)[:lens[i]]
            weak_sub_wav = apply_augmentation(args, weak_augmentation_list, sub_wav).to(args.device)

            with torch.no_grad():
                if teacher_model:
                    weak_outputs = teacher_model(weak_sub_wav).logits
                else:
                    weak_outputs = model(weak_sub_wav).logits

            weak_probs = F.softmax(weak_outputs, dim=-1)
            confidence, _ = torch.max(weak_probs, dim=-1, keepdim=True)
            weak_max_idx = torch.argmax(weak_probs, dim=-1, keepdim=True)
            weak_one_hots = torch.FloatTensor(weak_probs.shape).zero_().to(args.device).scatter(2, weak_max_idx, 1)
            non_blank = torch.where(weak_max_idx != 0, 1, 0).bool()

            selected_frames = non_blank & torch.where(confidence > args.prob_threshold, 1, 0).bool()
            selected_frames = selected_frames.expand_as(weak_probs)

            strong_sub_wav = apply_augmentation(args, strong_augmentation_list, sub_wav).to(args.device)
            strong_outputs = model(strong_sub_wav).logits

            for strong_output, weak_one_hot, selected_frame in zip(strong_outputs, weak_one_hots, selected_frames): # element-wise loss in batch
                cr_loss = ce_loss(
                    strong_output * selected_frame,
                    (weak_one_hot * selected_frame).detach()
                ) / (len(wavs) * len(strong_outputs))
                cr_loss.backward(retain_graph=True)
            del sub_wav, weak_sub_wav, weak_probs, confidence, weak_max_idx, non_blank, selected_frames, strong_sub_wav, strong_outputs

    if "em_joint" in args.method:
        for i, wav in wavs:
            for sub_wav in torch.chunk(wav[:lens[i]], chunks=args.n_neighbors + 2, dim=-1):
                sub_wav = sub_wav.unsqueeze(0)
                log_prob_tensor = model(sub_wav).logits
                max_log_probs, _ = torch.max(log_prob_tensor, dim=-1, keepdim=False)

                if args.certain_only:
                    probs = torch.softmax(log_prob_tensor, dim=-1)
                    confidence, _ = torch.max(probs, dim=-1, keepdim=True)
                    selected_tokens = torch.where(confidence > args.prob_threshold, 1, 0).bool()
                    max_log_probs = selected_tokens * max_log_probs

                if args.not_blank:
                    predicted_ids = torch.argmax(log_prob_tensor, dim=-1)
                    non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
                    max_log_probs = non_blank * max_log_probs

                sum_log_probs = torch.sum(max_log_probs, dim=-1)

                nll_loss = - sum_log_probs.mean()
                (nll_loss / (len(wavs) * (args.n_neighbors + 2))).backward()

                del sub_wav, log_prob_tensor, max_log_probs, sum_log_probs, nll_loss
                gc.collect()
                torch.cuda.empty_cache()

    if "gce" in args.method:
        q = 0.7
        for i, wav in enumerate(wavs):
            for sub_wav in torch.chunk(wav[:lens[i]], chunks=args.n_neighbors, dim=-1):
                sub_wav = sub_wav.unsqueeze(0)
                log_prob_tensor = model(sub_wav).logits

                probs = torch.softmax(log_prob_tensor / args.temp, dim=-1)
                max_probs, _ = torch.max(probs, dim=-1, keepdim=False)

                if args.certain_only:
                    confidence, _ = torch.max(torch.softmax(log_prob_tensor, dim=-1), dim=-1, keepdim=True)
                    selected_tokens = torch.where(confidence > args.prob_threshold, 1, 0).squeeze(2).detach()
                    max_probs = selected_tokens * max_probs

                if args.not_blank:
                    predicted_ids = torch.argmax(probs, dim=-1)
                    non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
                    max_probs = non_blank * max_probs

                gce_loss = torch.mean((1 - max_probs ** q) / q, dim=-1)
                (gce_loss / (len(wavs) * args.n_neighbors)).backward()

                del sub_wav, log_prob_tensor, max_probs, gce_loss
                gc.collect()
                torch.cuda.empty_cache()

    if 'ctc' in args.method:
        import json
        f = open('vocab.json')
        vocab = json.load(f)

        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)
            outputs = model(wav).logits
            ctc_loss = pseudo_labeling_loss(outputs, vocab, processor)

            (ctc_loss / len(wavs)).backward()

    if 'mixup' in args.method:
        from itertools import combinations
        combs = list(combinations(range(len(wavs)), 2))

        import json
        f = open('vocab.json')
        vocab = json.load(f)

        for i, j in combs:
            mix_ratio = Beta(torch.FloatTensor([2]), torch.FloatTensor([2])).sample().item()

            wav_i = wavs[i][:lens[i]].unsqueeze(0)
            wav_j = wavs[j][:lens[j]].unsqueeze(0)
            output_i = model(wav_i).logits
            output_j = model(wav_j).logits

            predicted_ids_i = torch.argmax(output_i, dim=-1)
            transcription_i = processor.batch_decode(predicted_ids_i)[0]
            predicted_ids_j = torch.argmax(output_j, dim=-1)
            transcription_j = processor.batch_decode(predicted_ids_j)[0]

            mixed_wav = (mix_ratio) * wavs[i].unsqueeze(0) + (1 - mix_ratio) * wavs[j].unsqueeze(0)
            mixed_output = model(mixed_wav).logits

            ctc_loss_i = get_pl_loss(mixed_output, transcription_i, vocab)
            ctc_loss_j = get_pl_loss(mixed_output, transcription_j, vocab)

            ((mix_ratio * ctc_loss_i + (1 - mix_ratio) * ctc_loss_j) / len(combs)).backward()

    if 'beam_search_max' in args.method or 'beam_search_all' in args.method:
        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)
            outputs = model(wav).logits
            predicted_ids = torch.argmax(outputs, dim=-1)
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()

            import json
            f = open('vocab.json')
            vocab = json.load(f)
            ctc_loss = nn.CTCLoss(blank=0, zero_infinity=False)

            if 'beam_search_max' in args.method:
                # beam_search_output = PROCESSOR_WITH_LM.decode(logits=outputs.squeeze(0).detach().cpu().numpy(), beam_width=1, output_word_offsets=True)

                import time
                current = time.time()

                if args.not_blank:
                    criterion = nn.CrossEntropyLoss(ignore_index=0)
                else:
                    criterion = nn.CrossEntropyLoss()

                logits = outputs.squeeze(0).detach().cpu().numpy()
                hotword_scorer = HotwordScorer.build_scorer(None, weight=DEFAULT_HOTWORD_WEIGHT)
                PROCESSOR_WITH_LM.decoder._check_logits_dimension(logits)

                # prepare hotword input
                # make sure we have log probs as input
                # if math.isclose(logits.sum(axis=1).mean(), 1):
                #     # input looks like probabilities, so take log
                #     logits = np.log(np.clip(logits, MIN_TOKEN_CLIP_P, 1))
                # else:
                #     # convert logits into log probs
                #     logits = np.clip(_log_softmax(logits, axis=1), np.log(MIN_TOKEN_CLIP_P), 0)

                beam_search_output = decode_beams_ctc(
                    PROCESSOR_WITH_LM.decoder,
                    logits=logits,
                    beam_width=1,
                    beam_prune_logp=DEFAULT_PRUNE_LOGP,
                    token_min_logp=DEFAULT_MIN_TOKEN_LOGP,
                    prune_history=DEFAULT_PRUNE_BEAMS,
                    hotword_scorer=hotword_scorer,
                    lm_start_state=None,
                )
                char_history = [*beam_search_output[0][-1]]
                char_history = ["<pad>" if char == "|" or char == " " else char for char in char_history]
                char_history = torch.tensor([vocab[char] for char in char_history]).to(args.device)

                print(f"time.time() - current: {time.time() - current}")

                if args.certain_only:
                    selected_frame = []
                    for frame_idx, (output, char_idx) in enumerate(zip(outputs.squeeze(0), char_history)):
                        probs = torch.softmax(output, dim=-1)
                        if probs[char_idx] > args.prob_threshold:
                            selected_frame.append(frame_idx)

                    outputs, char_history = outputs.squeeze(0)[selected_frame].unsqueeze(0), char_history[selected_frame]

                loss = criterion(outputs.squeeze(0) / args.temp, char_history)
                (loss / len(wavs)).backward(retain_graph=True)

            if 'beam_search_all' in args.method:
                if args.not_blank:
                    criterion = nn.CrossEntropyLoss(ignore_index=0)
                else:
                    criterion = nn.CrossEntropyLoss()

                logits = outputs.squeeze(0).detach().cpu().numpy()
                hotword_scorer = HotwordScorer.build_scorer(None, weight=DEFAULT_HOTWORD_WEIGHT)
                PROCESSOR_WITH_LM.decoder._check_logits_dimension(logits)

                # # prepare hotword input
                # # make sure we have log probs as input
                # if math.isclose(logits.sum(axis=1).mean(), 1):
                #     # input looks like probabilities, so take log
                #     logits = np.log(np.clip(logits, MIN_TOKEN_CLIP_P, 1))
                # else:
                #     # convert logits into log probs
                #     logits = np.clip(_log_softmax(logits, axis=1), np.log(MIN_TOKEN_CLIP_P), 0)

                beam_search_outputs = decode_beams_ctc(
                    PROCESSOR_WITH_LM.decoder,
                    logits=logits,
                    beam_width=args.beam_width,
                    beam_prune_logp=DEFAULT_PRUNE_LOGP,
                    token_min_logp=DEFAULT_MIN_TOKEN_LOGP,
                    prune_history=DEFAULT_PRUNE_BEAMS,
                    hotword_scorer=hotword_scorer,
                    lm_start_state=None,
                )
                loss_weights = torch.softmax(torch.tensor([beam_search_output[-2] for beam_search_output in beam_search_outputs]), dim=-1)

                # char_histories = []
                # for beam_search_output in beam_search_outputs:
                #     char_history = [*beam_search_output[-1]]
                #     char_history = ["<pad>" if char == "|" or char == " " else char for char in char_history]
                #     char_history = F.one_hot(torch.tensor([vocab[char] for char in char_history]), num_classes=outputs.shape[-1])
                #     char_histories.append(char_history)
                # char_histories = torch.stack(char_histories, dim=0)
                # one_hots = torch.sum(loss_weights.view(-1, 1, 1) * char_histories, dim=0)

                # loss = criterion(outputs.squeeze(0) / args.temp, one_hots)
                # (loss / len(wavs)).backward(retain_graph=True)

                loss = 0
                for out_idx, beam_search_output in enumerate(beam_search_outputs):
                    char_history = [*beam_search_output[-1]]
                    char_history = ["<pad>" if char == "|" or char == " " else char for char in char_history]
                    char_history = torch.tensor([vocab[char] for char in char_history]).to(args.device)

                    loss += loss_weights[out_idx] * criterion(outputs.squeeze(0) / args.temp, char_history)
                    # (loss * loss_weights[out_idx] / len(wavs)).backward(retain_graph=True)
                (loss / len(wavs)).backward(retain_graph=True)
    if 'beam_search_negative_sampling' in args.method:
        if args.not_blank:
            criterion = nn.CrossEntropyLoss(ignore_index=0)
        else:
            criterion = nn.CrossEntropyLoss()

        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)
            outputs = model(wav).logits

            import json
            f = open('vocab.json')
            vocab = json.load(f)

            logits = outputs.squeeze(0).detach().cpu().numpy()
            hotword_scorer = HotwordScorer.build_scorer(None, weight=DEFAULT_HOTWORD_WEIGHT)
            PROCESSOR_WITH_LM.decoder._check_logits_dimension(logits)

            # prepare hotword input
            # make sure we have log probs as input
            # if math.isclose(logits.sum(axis=1).mean(), 1):
            #     # input looks like probabilities, so take log
            #     logits = np.log(np.clip(logits, MIN_TOKEN_CLIP_P, 1))
            # else:
            #     # convert logits into log probs
            #     logits = np.clip(_log_softmax(logits, axis=1), np.log(MIN_TOKEN_CLIP_P), 0)

            beam_search_outputs = decode_beams_ctc(
                PROCESSOR_WITH_LM.decoder,
                logits=logits,
                beam_width=args.beam_width,
                beam_prune_logp=DEFAULT_PRUNE_LOGP,
                token_min_logp=DEFAULT_MIN_TOKEN_LOGP,
                prune_history=DEFAULT_PRUNE_BEAMS,
                hotword_scorer=hotword_scorer,
                lm_start_state=None,
            )
            char_history = [*beam_search_outputs[0][-1]]
            char_history = ["<pad>" if char == "|" or char == " " else char for char in char_history]
            char_history = torch.tensor([vocab[char] for char in char_history]).to(args.device)

            if args.certain_only:
                selected_frame = []
                for frame_idx, (output, char_idx) in enumerate(zip(outputs.squeeze(0), char_history)):
                    probs = torch.softmax(output, dim=-1)
                    if probs[char_idx] > args.prob_threshold:
                        selected_frame.append(frame_idx)

                positive_outputs, positive_char_history = outputs.squeeze(0)[selected_frame].unsqueeze(0), char_history[selected_frame]
            else:
                positive_outputs, positive_char_history = outputs, char_history

            # # TODO: remove (only for debugging)
            # num_pos = 0
            # for hist in positive_char_history:
            #     if hist != 0:
            #         num_pos += 1
            # print(f"num_pos: {num_pos}")

            positive_loss = criterion(positive_outputs.squeeze(0) / args.temp, positive_char_history)
            (positive_loss / len(wavs)).backward(retain_graph=True)

            negative_loss = 0
            if args.negative_sampling_method == "random":
                for _ in range(args.num_negatives):
                    negative_char_history = torch.randint(high=len(vocab), size=(len(beam_search_outputs[0][-1]), )).to(args.device)
                    negative_mask = (negative_char_history != char_history) & (char_history != 0)

                    negative_outputs = []
                    for output, mask in zip(outputs.squeeze(0), negative_mask):
                        if mask:
                            negative_outputs.append(output)
                    # print(f"len(negative_mask): {len(negative_mask)}")
                    # print(f"len(negative_outputs): {len(negative_outputs)}")

                    if len(negative_outputs) > 0:
                        negative_outputs = torch.stack(negative_outputs).unsqueeze(0)
                        negative_char_history = torch.masked_select(negative_char_history, negative_mask)

                        negative_loss += -criterion(negative_outputs.squeeze(0) / args.temp, negative_char_history) / args.num_negatives
                        # (- loss / (args.num_negatives * len(wavs))).backward(retain_graph=True)
            elif args.negative_sampling_method == "beam_candidate":
                # for beam_search_output in beam_search_outputs:
                    # print(f"beam_search_output: {beam_search_output}")

                for out_idx in range(len(beam_search_outputs))[-args.num_negatives:]:
                    negative_char_history = [*beam_search_outputs[out_idx][-1]]

                    # print(f"negative_char_history_before: {''.join(negative_char_history)}")

                    negative_char_history = ["<pad>" if char == "|" or char == " " else char for char in negative_char_history]
                    negative_char_history = torch.tensor([vocab[char] for char in negative_char_history]).to(args.device)

                    negative_mask = (negative_char_history != char_history) & (char_history != 0)

                    # print(f"negative_mask: {negative_mask}")
                    # print(f"char_history: {char_history}")

                    negative_outputs = []
                    for output, mask in zip(outputs.squeeze(0), negative_mask):
                        if mask:
                            negative_outputs.append(output)
                    # print(f"out_idx: {out_idx}")
                    # print(f"len(negative_mask): {len(negative_mask)}")
                    # print(f"len(negative_outputs): {len(negative_outputs)}")
                    if len(negative_outputs) > 0:
                        negative_outputs = torch.stack(negative_outputs).unsqueeze(0)
                        negative_char_history = torch.masked_select(negative_char_history, negative_mask)

                        negative_loss += -criterion(negative_outputs.squeeze(0) / args.temp, negative_char_history) * (len(negative_char_history) / max(1, len(positive_char_history)))

                        # negative_loss += -criterion(negative_outputs.squeeze(0) / args.temp, negative_char_history) / args.num_negatives
                        # (- loss / (args.num_negatives * len(wavs))).backward(retain_graph=True)
            if torch.is_tensor(negative_loss):
                (args.ns_coef * negative_loss / len(wavs)).backward(retain_graph=True)
    if 'beam_em_mix' in args.method:
        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)
            outputs = model(wav).logits

            import json
            f = open('vocab.json')
            vocab = json.load(f)

            criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
            logits = outputs.squeeze(0).detach().cpu().numpy()
            hotword_scorer = HotwordScorer.build_scorer(None, weight=DEFAULT_HOTWORD_WEIGHT)

            # PROCESSOR_WITH_LM.decoder._check_logits_dimension(logits)
            # # prepare hotword input
            # # make sure we have log probs as input
            # if math.isclose(logits.sum(axis=1).mean(), 1):
            #     # input looks like probabilities, so take log
            #     logits = np.log(np.clip(logits, MIN_TOKEN_CLIP_P, 1))
            # else:
            #     # convert logits into log probs
            #     logits = np.clip(_log_softmax(logits, axis=1), np.log(MIN_TOKEN_CLIP_P), 0)

            beam_search_output = decode_beams_ctc(
                PROCESSOR_WITH_LM.decoder,
                logits=logits,
                beam_width=args.beam_width,
                beam_prune_logp=DEFAULT_PRUNE_LOGP,
                token_min_logp=DEFAULT_MIN_TOKEN_LOGP,
                prune_history=DEFAULT_PRUNE_BEAMS,
                hotword_scorer=hotword_scorer,
                lm_start_state=None,
            )
            char_history = [*beam_search_output[0][-1]]
            char_history = ["<pad>" if char == "|" or char == " " else char for char in char_history]
            char_history = torch.tensor([vocab[char] for char in char_history]).to(args.device)

            predicted_ids = torch.argmax(char_history, dim=-1)
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()

            num_classes = outputs.shape[-1]
            ENTROPY_MIN, ENTROPY_MAX = 0, num_classes * (- (1 / num_classes) * np.log(1 / num_classes))

            entropy = (softmax_entropy(outputs / args.temp, dim=-1) - ENTROPY_MIN) / (ENTROPY_MAX - ENTROPY_MIN)
            entropy = entropy.detach()

            pseudo_label_loss = criterion(outputs.squeeze(0) / args.temp, char_history)
            entropy_loss = softmax_entropy(outputs / args.temp)

            loss = (1 - entropy) * pseudo_label_loss + entropy * entropy_loss
            loss = loss[non_blank].mean()
            (loss / len(wavs)).backward(retain_graph=True)
    if 'diversity_maximization' in args.method:
        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)
            outputs = model(wav).logits
            predicted_ids = torch.argmax(outputs, dim=-1)
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
            probs = torch.softmax(outputs[non_blank] / args.temp, dim=-1)
            mean_prob = torch.mean(probs, dim=0)
            loss = torch.sum(mean_prob * torch.log(mean_prob))
            (args.dm_coef * loss / len(wavs)).backward(retain_graph=True)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()


def forward_and_adapt_attn(args, model, teacher_model, processor, optimizer, scheduler, wavs, lens, adapter=None, step_idx=None):
    greedy_searcher = S2SRNNGreedySearcher(model.mods.decoder.emb, model.mods.decoder.dec, model.mods.decoder.fc, **{"bos_index": model.mods.decoder.bos_index, "eos_index": model.mods.decoder.eos_index, "min_decode_ratio": model.mods.decoder.min_decode_ratio, "max_decode_ratio": model.mods.decoder.max_decode_ratio,},).to(args.device).train()
    optimizer.zero_grad()

    if "original" in args.method or "em_uncertainty" in args.method or "em_sparse" in args.method:
        for i, wav in enumerate(wavs):
            wav = wav.unsqueeze(0)[:lens[i]]
            log_prob_tensor = forward_attn(args, model, greedy_searcher, wav)

            predicted_ids = torch.argmax(log_prob_tensor, dim=-1)
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()

            if args.em_coef > 0:
                if "original" in args.method:
                    e_loss = softmax_entropy(log_prob_tensor / args.temp, dim=-1)[non_blank].mean(0).mean()
                elif "em_uncertainty" in  args.method:
                    frame_weight = F.normalize(torch.reciprocal(softmax_entropy(log_prob_tensor)), p=1, dim=-1).detach()
                    e_loss = torch.sum(frame_weight * softmax_entropy(log_prob_tensor / args.temp), dim=-1).mean()
                elif "em_sparse" in args.method:
                    selected_frame = torch.where(softmax_entropy(log_prob_tensor, dim=-1) < args.entropy_threshold, 1, 0).bool()
                    e_loss = softmax_entropy(log_prob_tensor / args.temp)[selected_frame].mean(0).mean()

                (args.em_coef / wavs.shape[0] * e_loss).backward(retain_graph=True)

            if 1 - args.em_coef > 0:
                c_loss = mcc_loss(log_prob_tensor / args.temp, reweight=args.reweight, class_num=1000)
                ((1 - args.em_coef) / wavs.shape[0] * c_loss).backward(retain_graph=True)

    if "cr" in args.method:
        weak_augmentation_list, strong_augmentation_list = get_augmentation(args)
        seq_loss = lambda x, y, z: speechbrain.nnet.losses.nll_loss(x, y, z, label_smoothing=0.1)

        weak_wavs = apply_augmentation(args, weak_augmentation_list, wavs).to(args.device)
        with torch.no_grad():
            if teacher_model:
                teacher_greedy_searcher = S2SRNNGreedySearcher(
                    teacher_model.mods.decoder.emb,
                    teacher_model.mods.decoder.dec,
                    teacher_model.mods.decoder.fc,
                    **{
                        "bos_index": teacher_model.mods.decoder.bos_index,
                        "eos_index": teacher_model.mods.decoder.eos_index,
                        "min_decode_ratio": teacher_model.mods.decoder.min_decode_ratio,
                        "max_decode_ratio": teacher_model.mods.decoder.max_decode_ratio,
                    },
                ).to(args.device).train()
                weak_outputs = forward_attn(args, teacher_model, teacher_greedy_searcher, weak_wavs)
            else:
                weak_outputs = forward_attn(args, model, greedy_searcher, weak_wavs)

        weak_probs = F.softmax(weak_outputs, dim=-1)
        confidence, _ = torch.max(weak_probs, dim=-1, keepdim=True)
        weak_max_idx = torch.argmax(weak_probs, dim=-1, keepdim=True)
        non_blank = torch.where(weak_max_idx != 0, 1, 0).bool()

        selected_frames = non_blank & torch.where(confidence > args.prob_threshold, 1, 0).bool()
        selected_frames = selected_frames.expand_as(weak_probs)

        strong_wavs = apply_augmentation(args, strong_augmentation_list, wavs).to(args.device)
        strong_outputs = forward_attn(args, model, greedy_searcher, strong_wavs, gt_wavs=weak_wavs)

        cr_loss = seq_loss(
            strong_outputs, torch.argmax(weak_probs, dim=-1).detach(), torch.ones(len(strong_outputs)).to(args.device)
        )
        cr_loss.backward()

    if "cr_feature" in args.method:
        weak_augmentation_list, strong_augmentation_list = get_augmentation(args)

        for i, sub_wav in enumerate(wavs):
            sub_wav = sub_wav[:lens[i]].unsqueeze(0)
            weak_wavs = sub_wav.clone()
            l1_loss = nn.L1Loss()

            with torch.no_grad():
                if teacher_model:
                    teacher_greedy_searcher = S2SRNNGreedySearcher(
                        teacher_model.mods.decoder.emb,
                        teacher_model.mods.decoder.dec,
                        teacher_model.mods.decoder.fc,
                        **{
                            "bos_index": teacher_model.mods.decoder.bos_index,
                            "eos_index": teacher_model.mods.decoder.eos_index,
                            "min_decode_ratio": teacher_model.mods.decoder.min_decode_ratio,
                            "max_decode_ratio": teacher_model.mods.decoder.max_decode_ratio,
                        },
                    ).to(args.device).train()
                    weak_enc_states = teacher_model.encode_batch(weak_wavs, wav_lens=torch.ones(len(weak_wavs)).to(args.device))
                else:
                    weak_enc_states = model.encode_batch(weak_wavs, wav_lens=torch.ones(len(weak_wavs)).to(args.device))

            for _ in range(args.num_augs):
                strong_wavs = apply_augmentation(args, strong_augmentation_list, sub_wav).to(args.device).clone()
                strong_enc_states = model.encode_batch(strong_wavs, wav_lens=torch.ones(len(strong_wavs)).to(args.device))
                cr_feature_loss = l1_loss(strong_enc_states, weak_enc_states.detach())
                (cr_feature_loss / (args.num_augs * len(wavs))).backward()

    if "da" in args.method:
        with open("subspace_full.pkl", "rb") as f:
            source_subspace = pickle.load(f).float().to(args.device).transpose(0, 1) # D x d
        enc_states = model.encode_batch(wavs, wav_lens=torch.ones(len(wavs)).to(args.device)) # B x D x C
        enc_tensor = enc_states.view(-1, enc_states.shape[-1]).repeat(int(np.ceil(128 / (enc_states.shape[0] * enc_states.shape[1]))), 1) # (B * D) x C
        target_pca = PCA(n_components=128).fit(enc_tensor.detach().cpu())
        target_subspace = torch.FloatTensor(target_pca.components_).to(args.device).transpose(0, 1) # D x d

        if step_idx == 0:
            adapter.weight.data = torch.mm(target_subspace.transpose(0, 1), source_subspace) # d x d

        aligned_encoder_states = torch.mm(adapter(torch.mm(enc_states.view(-1, enc_states.shape[-1]), target_subspace)), source_subspace.transpose(0, 1)).unsqueeze(0) # 
        aligned_encoder_lens = torch.tensor([aligned_encoder_states.shape[1]]).to(args.device)

        log_probs_lst = []
        device = aligned_encoder_states.device
        batch_size = aligned_encoder_states.shape[0]
        memory = greedy_searcher.reset_mem(batch_size, device=device)
        inp_tokens = (aligned_encoder_states.new_zeros(batch_size).fill_(greedy_searcher.bos_index).long())
        max_decode_steps = int(aligned_encoder_states.shape[1] * greedy_searcher.max_decode_ratio)
        for _ in range(max_decode_steps):
            log_probs, memory, _ = greedy_searcher.forward_step(
                inp_tokens, memory, aligned_encoder_states, aligned_encoder_lens
            )
            log_probs_lst.append(log_probs)
            inp_tokens = log_probs.argmax(dim=-1)
        log_prob_tensor = torch.stack(log_probs_lst, dim=1)
        mse_loss += F.mse_loss(adapter(target_subspace), source_subspace, reduction='sum') # L_{\Phi}
        mse_loss.backward()

        e_loss = softmax_entropy(log_prob_tensor / args.temp, dim=-1).mean(0).mean()
        e_loss.backward()

    if "em_joint" in args.method:
        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)
            log_prob_tensor = forward_attn(args, model, greedy_searcher, wav)
            max_log_probs, _ = torch.max(log_prob_tensor, dim=-1, keepdim=False)

            if args.certain_only:
                probs = torch.softmax(log_prob_tensor, dim=-1)
                confidence, _ = torch.max(probs, dim=-1, keepdim=True)
                selected_tokens = torch.where(confidence > args.prob_threshold, 1, 0).bool()
                max_log_probs = selected_tokens * max_log_probs

            if args.not_blank:
                predicted_ids = torch.argmax(log_prob_tensor, dim=-1)
                non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
                max_log_probs = non_blank * max_log_probs

            sum_log_probs = torch.sum(max_log_probs, dim=-1)

            nll_loss = - sum_log_probs.mean()
            (nll_loss / len(wavs)).backward()

    if "p_logp" in args.method:
        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)
            log_prob_tensor = forward_attn(args, model, greedy_searcher, wav)
            prob_tensor = torch.softmax(log_prob_tensor, dim=-1)

            max_probs, _ = torch.max(prob_tensor, dim=-1, keepdim=False)
            prod_probs = torch.prod(max_probs, dim=-1)

            max_log_probs, _ = torch.max(log_prob_tensor, dim=-1, keepdim=False)
            sum_log_probs = torch.sum(max_log_probs, dim=-1)

            p_logp_loss = - prod_probs * sum_log_probs
            p_logp_loss = p_logp_loss.mean()
            (p_logp_loss / len(wavs)).backward()

    if "contrastive_temporal" in args.method:
        weak_augmentation_list, strong_augmentation_list = get_augmentation(args)
        info_nce_loss = InfoNCE()
        num_chunks = 5

        weak_wavs = apply_augmentation(args, strong_augmentation_list, wavs).to(args.device)
        strong_wavs = apply_augmentation(args, strong_augmentation_list, wavs).to(args.device)

        weak_chunks = torch.chunk(weak_wavs, num_chunks, dim=-1)[:-1]
        weak_chunks_pad = pad_sequence(weak_chunks, batch_first=True).squeeze(1)
        weak_lens = torch.tensor([float(weak_chunk.shape[-1]) for weak_chunk in weak_chunks])
        weak_enc = model.encode_batch(weak_chunks_pad, wav_lens=weak_lens)
        weak_enc_pool = torch.mean(weak_enc, dim=1)

        strong_chunks = torch.chunk(strong_wavs, num_chunks, dim=-1)[:-1]
        strong_chunks_pad = pad_sequence(strong_chunks, batch_first=True).squeeze(1)
        strong_lens = torch.tensor([float(strong_chunk.shape[-1]) for strong_chunk in strong_chunks])
        strong_enc = model.encode_batch(strong_chunks_pad, wav_lens=strong_lens)
        strong_enc_pool = torch.mean(strong_enc, dim=1)

        nce_loss = info_nce_loss(weak_enc_pool, strong_enc_pool)
        nce_loss.backward()

    if "em_aug" in args.method:
        num_augs = args.num_augs
        _, strong_augmentation_list = get_augmentation(args)
 
        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)
            for i in range(num_augs):
                if i > 0:
                    aug_wav = apply_augmentation(args, strong_augmentation_list, wav).to(args.device)
                else:
                    aug_wav = wav

                log_prob_tensor = forward_attn(args, model, greedy_searcher, aug_wav)
                predicted_ids = torch.argmax(log_prob_tensor, dim=-1)

                mean_prob = torch.mean(torch.softmax(log_prob_tensor, dim=-1), dim=0, keepdim=True)
                mean_log_prob = torch.mean(torch.log_softmax(log_prob_tensor, dim=-1), dim=0, keepdim=True)

                e_loss = - torch.sum(mean_prob * mean_log_prob, dim=-1).mean()
                (e_loss / (len(wavs) * num_augs)).backward()
    
    if args.use_memory_queue:
        global memory_queue, HASH_COUNTER

        # get current gradient
        current_grad_dict = dict()
        for np, p in model.named_parameters():
            current_grad_dict[np] = p.grad if p.grad == None else p.grad.cpu().clone()

        # search wavs to adapt
        probs = torch.softmax(log_prob_tensor, dim=-1)
        _, _, hash_values_to_adapt = get_instance_from_queue(args, args.queue_method, wavs, probs)

        if len(hash_values_to_adapt) > 0:
            grad_dict_list = [db[hash_value_to_adapt] for hash_value_to_adapt in hash_values_to_adapt]
            cumulated_grad = dict()
            for np, _ in model.named_parameters():
                grads_np = [grad_dict[np] for grad_dict in grad_dict_list]
                cumulated_grad[np] = sum(grads_np) if not None in grads_np else None

            denominator = len(wavs) + len(hash_values_to_adapt)
            for np, p in model.named_parameters():  
                if p.grad == None:
                    continue
                p.grad = p.grad * len(wavs) / denominator + cumulated_grad[np].to(args.device) / denominator

        for wav, len, prob in zip(wavs, lens, probs):
            # dequeue
            while len(memory_queue) >= args.queue_size:
                wav_to_remove, _, hash_value_to_remove = memory_queue.popleft()
                del db[hash_value_to_remove]

            # enqueue
            hash_value = str(HASH_COUNTER)
            memory_queue.append((wav[:len].cpu().detach(), prob.cpu().detach(), hash_value))
            db[hash_value] = current_grad_dict
            HASH_COUNTER += 1

            ns_loss = non_saturating_loss(log_prob_tensor)
            (ns_loss / len(wavs)).backward()

    optimizer.step()
    if scheduler is not None:
        scheduler.step()


def forward_and_adapt_trans(args, model, teacher_model, processor, optimizer, scheduler, wavs, lens):
    optimizer.zero_grad()

    if "original" in args.method or "em_uncertainty" in args.method or "em_sparse" in args.method:
        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)

            import time
            current = time.time()

            log_prob_tensor = forward_trans(args, model, wav, torch.tensor([lens[i]]).to(wav.device), gt_wavs=None)

            # decoded_outputs = decode_beams_trans(beam_search_decoder, encoder_output, logitlen)

            # print(f"time.time() - current: {time.time() - current}")

            if args.em_coef > 0:
                if "original" in args.method:
                    e_loss = softmax_entropy(log_prob_tensor / args.temp, dim=-1).mean(0).mean()
                elif "em_uncertainty" in args.method:
                    frame_weight = F.normalize(torch.reciprocal(softmax_entropy(log_prob_tensor)), p=1, dim=-1).detach()
                    e_loss = torch.sum(frame_weight * softmax_entropy(log_prob_tensor / args.temp), dim=-1).mean()
                elif "em_sparse" in args.method:
                    selected_frame = torch.where(softmax_entropy(log_prob_tensor, dim=-1) < args.entropy_threshold, 1, 0).bool()
                    e_loss = softmax_entropy(log_prob_tensor / args.temp)[selected_frame].mean(0).mean()
                ((args.em_coef / len(wavs)) * e_loss).backward(retain_graph=True)

            if 1 - args.em_coef > 0:
                c_loss = mcc_loss(log_prob_tensor / args.temp, reweight=args.reweight, class_num=1000)
                (((1 - args.em_coef) / len(wavs)) * c_loss).backward(retain_graph=True)

            print(f"time.time() - current: {time.time() - current}")

    if "cr" in args.method:
        weak_augmentation_list, strong_augmentation_list = get_augmentation(args)
        ctc_loss = CTCLoss(num_classes=1000)

        weak_wavs = apply_augmentation(args, weak_augmentation_list, wavs).to(args.device)
        with torch.no_grad():
            if teacher_model:
                weak_outputs = forward_trans(args, teacher_model, weak_wavs, lens, gt_wavs=None)
            else:
                weak_outputs = forward_trans(args, model, weak_wavs, lens, gt_wavs=None)

        weak_probs = F.softmax(weak_outputs, dim=-1)
        confidence, _ = torch.max(weak_probs, dim=-1, keepdim=True)

        weak_max_idx = torch.argmax(weak_probs, dim=-1, keepdim=True)
        weak_one_hots = torch.FloatTensor(weak_probs.shape).zero_().to(args.device).scatter(2, weak_max_idx, 1)
        non_blank = torch.where(weak_max_idx != model.decoding.decoding._blank_index, 1, 0).bool()

        selected_frames = non_blank & torch.where(confidence > args.prob_threshold, 1, 0).bool()
        selected_frames = selected_frames.expand_as(weak_probs)

        del weak_outputs, weak_probs, confidence, non_blank

        strong_wavs = apply_augmentation(args, strong_augmentation_list, wavs).to(args.device)
        strong_outputs = forward_trans(args, model, strong_wavs, lens, gt_wavs=weak_wavs)

        if strong_outputs.shape[1] > weak_one_hots.shape[1]:
            strong_outputs = strong_outputs[:, :weak_one_hots.shape[1], :]
        else:
            weak_one_hots = weak_one_hots[:, :strong_outputs.shape[1], :]
            selected_frames = selected_frames[:, :strong_outputs.shape[1], :]

        loss += ctc_loss(
            log_probs=torch.permute(strong_outputs, (1, 0, 2)),
            targets=weak_max_idx.squeeze(-1),
            input_lengths=torch.ones(len(strong_outputs)).to(args.device),
            target_lengths=torch.ones(len(strong_outputs)).to(args.device)
        )
        del strong_wavs, strong_outputs
    if "em_joint" in args.method:
        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)
            log_prob_tensor = forward_trans(args, model, wav, torch.tensor([lens[i]]).to(wav.device), gt_wavs=None)
            max_log_probs, _ = torch.max(log_prob_tensor, dim=-1, keepdim=False)

            if args.certain_only:
                probs = torch.softmax(log_prob_tensor, dim=-1)
                confidence, _ = torch.max(probs, dim=-1, keepdim=True)
                selected_tokens = torch.where(confidence > args.prob_threshold, 1, 0).bool().detach()
                max_log_probs = selected_tokens * max_log_probs

            if args.not_blank:
                predicted_ids = torch.argmax(log_prob_tensor, dim=-1)
                non_blank = torch.where(predicted_ids != model.decoding.decoding._blank_index, 1, 0).bool().detach()
                max_log_probs = non_blank * max_log_probs

            sum_log_probs = torch.sum(max_log_probs, dim=-1)

            nll_loss = - sum_log_probs.mean()
            (nll_loss / len(wavs)).backward()
    if 'beam_search_max' in args.method or 'beam_search_all' in args.method or 'beam_search_em' in args.method:
        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)

            encoder_output, encoded_lengths = model(input_signal=wav, input_signal_length=torch.tensor([lens[i]]).to(wav.device))
            encoder_output = encoder_output.transpose(1, 2)
            logitlen = encoded_lengths

            beam_search_decoder = BeamRNNTInfer(
                model.decoding.decoding.decoder.to(args.device),
                model.decoding.decoding.joint.to(args.device),
                beam_size=args.beam_width,
                return_best_hypothesis=False,
            )

            import time
            current = time.time()

            decoded_output = decode_beams_trans(beam_search_decoder, encoder_output, logitlen)[0]
            outputs = torch.stack(decoded_output.logit_list, dim=0).unsqueeze(0)

            print(f"beam search time: {time.time() - current}")

            if args.not_blank:
                criterion = nn.CrossEntropyLoss(ignore_index=model.decoding.decoding._blank_index)
            else:
                criterion = nn.CrossEntropyLoss()

            if 'beam_search_max' in args.method:
                char_history = torch.stack(decoded_output.token_list, dim=0)

                if args.certain_only:
                    selected_frame = []
                    for frame_idx, (output, char_idx) in enumerate(zip(outputs.squeeze(0), char_history)):
                        probs = torch.softmax(output, dim=-1)
                        if probs[char_idx] > args.prob_threshold:
                            selected_frame.append(frame_idx)
                    outputs, char_history = outputs.squeeze(0)[selected_frame].unsqueeze(0), char_history[selected_frame]

                if len(selected_frame) > 0:
                    loss = criterion(outputs.squeeze(0) / args.temp, char_history)
                    (loss / len(wavs)).backward(retain_graph=True)

            if 'beam_search_em' in args.method:
                char_history = torch.stack(decoded_output.token_list, dim=0).unsqueeze(0)
                non_blank = torch.where(char_history != model.decoding.decoding._blank_index, 1, 0).bool()

                if args.not_blank:
                    e_loss = softmax_entropy(outputs / args.temp)[non_blank].mean(0).mean()
                else:
                    e_loss = softmax_entropy(outputs / args.temp).mean(0).mean()
                (e_loss / len(wavs)).backward(retain_graph=True)


    optimizer.step()
    if scheduler is not None:
        scheduler.step()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args):
    dataset_name = args.dataset_name
    dataset_dir = args.dataset_dir
    split = args.split
    batch_size = args.batch_size
    extra_noise = args.extra_noise
    noise_type = args.noise_type
    sample_rate = args.sample_rate

    optimizer = args.optimizer
    lr = args.lr
    scheduler = args.scheduler
    steps = args.steps
    train_params = args.train_params
    bias_only = args.bias_only
    episodic = args.episodic

    teacher_student = args.teacher_student
    momentum = args.momentum

    stochastic_restoration = args.stochastic_restoration
    restore_prob = args.restore_prob

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    global logger
    logger = get_logger(args)
    logger.info(OmegaConf.to_yaml(args))

    dataset = load_dataset(split, dataset_name, dataset_dir, batch_size, extra_noise, noise_type=noise_type)
    gt_texts, ori_transcriptions, transcriptions_1, transcriptions_3, transcriptions_5, transcriptions_10, transcriptions_20, transcriptions_40 = [], [], [], [], [], [], [], []

    original_model = get_model(args, original=True)
    model = get_model(args, original=False)

    use_memory_queue = args.use_memory_queue
    queue_size = args.queue_size
    # n_neighbors = args.n_neighbors
    if use_memory_queue:
        if isinstance(model, EncoderDecoderASR):
            assert steps == 1 and batch_size == 1

        global memory_queue, db, HASH_COUNTER
        memory_queue = deque([], maxlen=queue_size)
        time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if not os.path.exists("grad_dict"):
            os.makedirs("grad_dict")
        db = shelve.open(f"grad_dict/grads_{time_string}.pkl", writeback=True)
        HASH_COUNTER = 0


    if isinstance(model, Wav2Vec2ForCTC): # ctc
        params, _ = collect_params_ctc(model, train_params, bias_only)
    elif isinstance(model, EncoderDecoderASR):
        params, _ = collect_params_attn(model, train_params, bias_only)
    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
        params, _ = collect_params_trans(model, train_params, bias_only)
    optimizer, scheduler = get_optimizer(params, opt_name=args.optimizer, lr=lr, scheduler=args.scheduler)

    teacher_model = get_model(args, original=False) if teacher_student else None
    processor = Wav2Vec2Processor.from_pretrained(args.asr, sampling_rate=sample_rate, return_attention_mask=True) if isinstance(model, Wav2Vec2ForCTC) else None

    if episodic:
        original_model_state, original_optimizer_state, original_scheduler_state = copy_model_and_optimizer(model, optimizer, scheduler)

    for batch_idx, batch in enumerate(dataset):
        # if batch_idx >= 100:
        #     break

        lens, wavs, texts, _ = batch

        if isinstance(model, Wav2Vec2ForCTC):
            wavs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding="longest").input_values.to(args.device)
        else:
            wavs = pad_sequence([torch.from_numpy(wav) for wav in wavs], batch_first=True).to(args.device)
        lens = lens.to(args.device)

        adapt_or_not = True
        with torch.no_grad():
            if (args.use_memory_queue or args.selective_adaptation) and not isinstance(model, EncoderDecoderASR):
                probs = []
                for i, wav in enumerate(wavs):
                    wav = wav.unsqueeze(0)
                    if isinstance(model, Wav2Vec2ForCTC):
                        logit = model(wav).logits
                    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
                        logit = forward_trans(args, model, wav, torch.tensor([lens[i]]).to(wav.device), gt_wavs=None)
                probs.append(torch.softmax(logit.squeeze(0), dim=-1))
                probs = pad_sequence(probs, batch_first=True).to(args.device)

                if args.selective_adaptation:
                    per_token_ood, _ = torch.max(probs, dim=-1)
                    per_token_odd = - per_token_ood
                    max_ood, _ = torch.max(per_token_odd, dim=-1) # max_ood per each instance
                    avg_max_ood = torch.mean(max_ood, dim=-1) # average max_ood per batch
                    if avg_max_ood < args.ood_threshold:
                        adapt_or_not = False

                if args.use_memory_queue:
                    wavs_to_adapt, lens_to_adapt = [], []

                    wavs_queue, lens_queue, _ = get_instance_from_queue(args, args.queue_method, wavs, probs)
                    for wav_queue, len_queue in zip(wavs_queue, lens_queue):
                        wavs_to_adapt.append(wav_queue)
                        lens_to_adapt.append(len_queue)
                    for wav in wavs:
                        wavs_to_adapt.append(wav)
                        lens_to_adapt.append(len(wav))

                    wavs_to_adapt = pad_sequence(wavs_to_adapt, batch_first=True)
                    lens_to_adapt = torch.tensor(lens_to_adapt).to(args.device)

                    gc.collect()
                    torch.cuda.empty_cache()

                else:
                    wavs_to_adapt = wavs
                    lens_to_adapt = lens
            else:
                wavs_to_adapt = wavs
                lens_to_adapt = lens

        gt_texts.extend(texts)
        ori_transcription = transcribe_batch(args, original_model, processor, wavs, lens)
        ori_transcriptions.extend(ori_transcription)

        ori_wer = wer(list(texts), list(ori_transcription))
        logger.info(f"{batch_idx}/{len(dataset)}")
        logger.info(f"gt text: {list(texts)}")
        logger.info(f"original WER: {ori_wer}")
        logger.info(f"original text: {list(ori_transcription)}")

        if episodic:
            if "da" in args.method:
                model = deepcopy(original_model)
                if isinstance(model, Wav2Vec2ForCTC): # ctc
                    params, _ = collect_params_ctc(model, train_params, bias_only)
                elif isinstance(model, EncoderDecoderASR):
                    params, _ = collect_params_attn(model, train_params, bias_only)
                elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
                    params, _ = collect_params_trans(model, train_params, bias_only)
                optimizer, scheduler = get_optimizer(params, opt_name=args.optimizer, lr=lr, scheduler=args.scheduler)
            else:
                model, optimizer, scheduler = load_model_and_optimizer(model, optimizer, scheduler, original_model_state, original_optimizer_state, original_scheduler_state)

        if "da" in args.method:
            adapter = nn.Linear(in_features=128, out_features=128, bias=False).requires_grad_(True).to(args.device)
            optimizer.add_param_group({'params': [p for p in adapter.parameters()]})
        else:
            adapter = None

        for step_idx in range(1, steps + 1):
            current = time.time()
            if adapt_or_not:
                if isinstance(model, Wav2Vec2ForCTC): # ctc
                    forward_and_adapt_ctc(args, model, teacher_model, processor, optimizer, scheduler, wavs_to_adapt, lens_to_adapt)
                elif isinstance(model, EncoderDecoderASR): # attention-based encoder-decoder
                    forward_and_adapt_attn(args, model, teacher_model, processor, optimizer, scheduler, wavs_to_adapt, lens_to_adapt, adapter=adapter, step_idx=step_idx)
                elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel): # transducer
                    forward_and_adapt_trans(args, model, teacher_model, processor, optimizer, scheduler, wavs_to_adapt, lens_to_adapt)

            if step_idx in [1, 3, 5, 10, 20, 40]:
                transcription = transcribe_batch(args, model, processor, wavs, lens)
                transcription_list = eval(f"transcriptions_{step_idx}")
                transcription_list.extend(transcription)

                ada_wer = wer(list(texts), list(transcription))
                logger.info(f"adapt-{step_idx} WER: {ada_wer}")
                logger.info(f"adapt-{step_idx} text: {' '.join(list(transcription))}")

            gc.collect()
            torch.cuda.empty_cache()

        if args.use_memory_queue and adapt_or_not and not isinstance(model, EncoderDecoderASR):
            for wav, prob in zip(wavs, probs):
                while len(memory_queue) >= args.queue_size:
                    wav_to_remove = memory_queue.popleft()[0]
                memory_queue.append((wav.cpu().detach(), prob.cpu().detach(), str(hash(wav.cpu().detach()))))

        if stochastic_restoration:
            for model_param, original_param in zip(model.parameters(), original_model.parameters()):
                restore = np.random.binomial(n=1, p=restore_prob, size=1)[0]
                with torch.no_grad():
                    model_param.copy_((1 - restore) * model_param + restore * original_param)

        if teacher_student:
            for teacher_param, model_param in zip(teacher_model.parameters(), model.parameters()):
                with torch.no_grad():
                    teacher_param.copy_(momentum * teacher_param + (1 - momentum) * model_param)

        logger.info("\n")

    logger.info(OmegaConf.to_yaml(args))
    logger.info(f"number of data : {len(dataset)}")
    logger.info(f"original WER: {wer(gt_texts, ori_transcriptions)}")
    for step_idx in [1, 3, 5, 10, 20, 40]:
        if step_idx <= steps:
            transcription_list = eval(f"transcriptions_{step_idx}")
            logger.info(f"TTA-{step_idx}: {wer(gt_texts, transcription_list)}")



if __name__ == '__main__':
    PROCESSOR_WITH_LM = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")
    main()