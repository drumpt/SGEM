import time
import os
import random
import gc
import logging
import shelve
from copy import deepcopy
from datetime import datetime
from collections import deque

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Beta
from info_nce import InfoNCE
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
import speechbrain
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.lobes.augment import TimeDomainSpecAugment
from speechbrain.decoders.seq2seq import S2SRNNGreedySearcher, S2SBaseSearcher
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.parts.submodules import rnnt_greedy_decoding
from nemo.collections.asr.parts.submodules.rnnt_beam_decoding import BeamRNNTInfer
from pyctcdecode.language_model import HotwordScorer
from pyctcdecode.constants import (
    # DEFAULT_BEAM_WIDTH,
    DEFAULT_HOTWORD_WEIGHT,
    DEFAULT_MIN_TOKEN_LOGP,
    DEFAULT_PRUNE_LOGP,
    DEFAULT_PRUNE_BEAMS
)
from audio_augmentations import *
from jiwer import wer

from data import load_dataset
from forward import transcribe_batch, forward_batch, decode_beams_ctc, decode_beams_attn, decode_beams_trans
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
    # CTC-based models
    if args.asr == "facebook/wav2vec2-base-960h":
        model = Wav2Vec2ForCTC.from_pretrained(args.asr).requires_grad_(True).eval()
        if 'cuda' in args.device:
            model = model.cuda()

    # attention-based models
    elif args.asr == "speechbrain/asr-crdnn-rnnlm-librispeech":
        model = EncoderDecoderASR.from_hparams(args.asr, run_opts={"device": args.device}).requires_grad_(True).eval()
    elif args.asr == "speechbrain/asr-crdnn-transformerlm-librispeech":
        model = EncoderDecoderASR.from_hparams(args.asr, run_opts={"device": args.device}).requires_grad_(True).eval()
    elif args.asr == "speechbrain/asr-transformer-transformerlm-librispeech":
        model = EncoderDecoderASR.from_hparams(args.asr, run_opts={"device": args.device}).requires_grad_(True).eval()
    elif args.asr == "speechbrain/asr-conformersmall-transformerlm-librispeech":
        model = EncoderDecoderASR.from_hparams(args.asr, run_opts={"device": args.device}).requires_grad_(True).eval()

    # transducers
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
    model.requires_grad_(False)
    return model


def freeze_norm_stats(model):
    for _, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            m.track_running_stats = False


def eval_except_for_rnn(model):
    model.eval()
    if isinstance(model, EncoderDecoderASR):
        for nm, m in model.named_modules():
            if 'rnn' in nm.lower() or 'lstm' in nm.lower():
                m.train()
                m.dropout = 0


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


def forward_and_adapt(args, model, processor, optimizer, scheduler, wavs, lens):
    optimizer.zero_grad()

    if "original" in args.method or "em_uncertainty" in args.method or "em_sparse" in args.method:
        for i, wav in enumerate(wavs):
            wav = wav.unsqueeze(0)[:lens[i]]
            outputs = forward_batch(args, model, wav, torch.FloatTensor([lens[i]]).to(wav.device))
            predicted_ids = torch.argmax(outputs, dim=-1)
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()

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
                (args.em_coef * e_loss / (len(wavs))).backward(retain_graph=True)

            if 1 - args.em_coef > 0:
                c_loss = mcc_loss(outputs / args.temp, args.reweight)
                ((1 - args.em_coef) * c_loss / (len(wavs))).backward(retain_graph=True)
    if "em_dropout" in args.method:
        for i, wav in enumerate(wavs):
            wav = wav.unsqueeze(0)[:lens[i]]
            e_loss = 0
            model.train()
            NUM_DROPOUTS = 5
            for _ in range(NUM_DROPOUTS):
                outputs = forward_batch(args, model, wav, torch.FloatTensor([lens[i]]).to(wav.device))
                predicted_ids = torch.argmax(outputs, dim=-1)
                non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
                e_loss += (softmax_entropy(outputs / args.temp)[non_blank].mean(0).mean()) / NUM_DROPOUTS
            model.eval()
            (args.em_coef * e_loss / (len(wavs))).backward(retain_graph=True)
    if "cr" in args.method:
        weak_augmentation_list, strong_augmentation_list = get_augmentation(args)

        ce_loss = nn.CrossEntropyLoss()
        for i, sub_wav in enumerate(wavs): # element-wise iteration
            sub_wav = sub_wav.unsqueeze(0)[:lens[i]]
            weak_sub_wav = apply_augmentation(args, weak_augmentation_list, sub_wav).to(args.device)
            with torch.no_grad():
                if teacher_model:
                    weak_outputs = forward_batch(args, teacher_model, weak_sub_wav, torch.FloatTensor([lens[i]]).to(weak_sub_wav.device))
                else:
                    weak_outputs = forward_batch(args, model, weak_sub_wav, torch.FloatTensor([lens[i]]).to(weak_sub_wav.device))

            weak_probs = F.softmax(weak_outputs, dim=-1)
            confidence, _ = torch.max(weak_probs, dim=-1, keepdim=True)
            weak_max_idx = torch.argmax(weak_probs, dim=-1, keepdim=True)
            weak_one_hots = torch.FloatTensor(weak_probs.shape).zero_().to(args.device).scatter(2, weak_max_idx, 1)
            non_blank = torch.where(weak_max_idx != 0, 1, 0).bool()

            selected_frames = non_blank & torch.where(confidence > args.prob_threshold, 1, 0).bool()
            selected_frames = selected_frames.expand_as(weak_probs)

            strong_sub_wav = apply_augmentation(args, strong_augmentation_list, sub_wav).to(args.device)
            strong_outputs = forward_batch(args, model, strong_sub_wav, torch.FloatTensor([lens[i]]).to(strong_sub_wav.device))
            for strong_output, weak_one_hot, selected_frame in zip(strong_outputs, weak_one_hots, selected_frames): # element-wise loss in batch
                cr_loss = ce_loss(
                    strong_output * selected_frame,
                    (weak_one_hot * selected_frame).detach()
                ) / (len(wavs) * len(strong_outputs))
                cr_loss.backward(retain_graph=True)

            del sub_wav, weak_sub_wav, weak_probs, confidence, weak_max_idx, non_blank, selected_frames, strong_sub_wav, strong_outputs
    if "em_joint" in args.method:
        for i, wav in enumerate(wavs):
            wav = wav.unsqueeze(0)[:lens[i]]
            log_prob_tensor = forward_batch(args, model, wav, torch.FloatTensor([lens[i]]).to(wav.device))
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

            del wav, log_prob_tensor, max_log_probs, sum_log_probs, nll_loss
    if "gce" in args.method:
        q = 0.7
        for i, wav in enumerate(wavs):
            wav = wav.unsqueeze(0)[:lens[i]]
            log_prob_tensor = forward_batch(args, model, wav, torch.FloatTensor([lens[i]]).to(wav.device))
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
            (gce_loss / len(wavs)).backward()

            del wav, log_prob_tensor, max_probs, gce_loss
    if 'ctc' in args.method:
        for i, wav in enumerate(wavs):
            import json
            f = open('vocab.json')
            vocab = json.load(f)

            wav = wav[:lens[i]].unsqueeze(0)
            outputs = forward_batch(args, model, wav, torch.FloatTensor([lens[i]]).to(wav.device))
            ctc_loss = pseudo_labeling_loss(outputs, vocab, processor)

            (ctc_loss / len(wavs)).backward()
    if 'beam_search_max' in args.method or 'beam_search_all' in args.method:
        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)
            outputs = forward_batch(args, model, wav, torch.FloatTensor([lens[i]]).to(wav.device))
            # predicted_ids = torch.argmax(outputs, dim=-1)
            # non_blank = torch.where(predicted_ids != 0, 1, 0).bool()

            # TODO: for ctc-based model
            if args.not_blank:
                criterion = nn.CrossEntropyLoss(ignore_index=0)
            else:
                criterion = nn.CrossEntropyLoss()

            logits = outputs.squeeze(0).detach().cpu().numpy()
            # logits = outputs.squeeze(0).cpu()
            hotword_scorer = HotwordScorer.build_scorer(None, weight=DEFAULT_HOTWORD_WEIGHT)
            PROCESSOR_WITH_LM.decoder._check_logits_dimension(logits)

            # import time
            # current = time.time()

            idx_history, lm_logits = decode_beams_ctc(
                PROCESSOR_WITH_LM.decoder,
                logits=logits,
                beam_width=20,
                beam_prune_logp=DEFAULT_PRUNE_LOGP,
                token_min_logp=DEFAULT_MIN_TOKEN_LOGP,
                prune_history=DEFAULT_PRUNE_BEAMS,
                hotword_scorer=hotword_scorer,
                lm_start_state=None,
            )[0]
            # char_history = torch.tensor([*beam_search_output[0][-1]]).to(args.device)
            # print(f"beam_search_output: {beam_search_output}")
            # print(f"time.time() - current: {time.time() - current}")

            print(f"outputs.device: {outputs.device}")
            print(f"torch.from_numpy(np.array(lm_logits)).unsqueeze(0).device: {torch.from_numpy(np.array(lm_logits)).unsqueeze(0).device}")

            combined_logits = outputs + torch.from_numpy(np.array(lm_logits)).unsqueeze(0).to(args.device)
            predicted_ids = torch.argmax(combined_logits, dim=-1)
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()

            e_loss = softmax_entropy(combined_logits / args.temp)[non_blank].mean(0).mean()
            (e_loss / (len(wavs))).backward(retain_graph=True)

            # TODO: for attention-based model
            # from speechbrain.decoders.seq2seq import S2SRNNBeamSearchLM
            # encoder_out = model.encode_batch(wav, torch.FloatTensor([lens[i]]).to(wav.device))

            # model.mods.decoder.topk=50
            # model.mods.decoder.beam_size=50

            # log_probs, hyps = decode_beams_attn(
            #     model.mods.decoder,
            #     encoder_out,
            #     torch.FloatTensor([lens[i]]).to(wav.device)
            # )
            # print(f"hyps.shape: {hyps.shape}")
            # print(f"log_probs: {log_probs}")


            # searcher = S2SRNNBeamSearchLM(
            #     embedding=model.mods.decoder.emb,
            #     decoder=model.mods.decoder.dec,
            #     linear=model.mods.decoder.fc,
            #     language_model=model.mods.decoder.lm,
            #     bos_index=model.mods.decoder.bos_index,
            #     eos_index=model.mods.decoder.eos_index,
            #     blank_index=model.mods.decoder.blank_index, # todo: change
            #     min_decode_ratio=model.mods.decoder.min_decode_ratio,
            #     max_decode_ratio=model.mods.decoder.max_decode_ratio,
            #     lm_weight=1,
            #     beam_size=20,
            #     topk=20,
            # ).to(args.device)
            # hyps, scores, log_probs = decode_beams_attn(searcher, encoder_out, torch.FloatTensor([lens[i]]).to(wav.device))

            # TODO: for transducers
            # encoder_output, encoded_lengths = model(input_signal=wav, input_signal_length=torch.FloatTensor([lens[i]]).to(wav.device))
            # encoder_output = encoder_output.transpose(1, 2)
            # logitlen = encoded_lengths

            # beam_search_decoder = BeamRNNTInfer(
            #     model.decoding.decoding.decoder.to(args.device),
            #     model.decoding.decoding.joint.to(args.device),
            #     beam_size=args.beam_width,
            #     return_best_hypothesis=False,
            # )
            # decoded_output = decode_beams_trans(beam_search_decoder, encoder_output, logitlen)[0]
            # outputs = torch.stack(decoded_output.logit_list, dim=0).unsqueeze(0)
            # print(f"outputs: {outputs}")

            # if 'beam_search_max' in args.method:
            #     if args.not_blank:
            #         criterion = nn.CrossEntropyLoss(ignore_index=0)
            #     else:
            #         criterion = nn.CrossEntropyLoss()

            #     logits = outputs.squeeze(0).detach().cpu().numpy()
            #     hotword_scorer = HotwordScorer.build_scorer(None, weight=DEFAULT_HOTWORD_WEIGHT)
            #     PROCESSOR_WITH_LM.decoder._check_logits_dimension(logits)

            #     beam_search_output = decode_beams_ctc(
            #         PROCESSOR_WITH_LM.decoder,
            #         logits=logits,
            #         beam_width=1,
            #         beam_prune_logp=DEFAULT_PRUNE_LOGP,
            #         token_min_logp=DEFAULT_MIN_TOKEN_LOGP,
            #         prune_history=DEFAULT_PRUNE_BEAMS,
            #         hotword_scorer=hotword_scorer,
            #         lm_start_state=None,
            #     )
            #     char_history = torch.tensor([*beam_search_output[0][-1]]).to(args.device)
            #     if args.certain_only:
            #         selected_frame = []
            #         for frame_idx, (output, char_idx) in enumerate(zip(outputs.squeeze(0), char_history)):
            #             probs = torch.softmax(output, dim=-1)
            #             if probs[char_idx] > args.prob_threshold:
            #                 selected_frame.append(frame_idx)

            #         outputs, char_history = outputs.squeeze(0)[selected_frame].unsqueeze(0), char_history[selected_frame]

            #     loss = criterion(outputs.squeeze(0) / args.temp, char_history)
            #     (loss / len(wavs)).backward(retain_graph=True)

            # if 'beam_search_all' in args.method:
            #     if args.not_blank:
            #         criterion = nn.CrossEntropyLoss(ignore_index=0)
            #     else:
            #         criterion = nn.CrossEntropyLoss()

            #     logits = outputs.squeeze(0).detach().cpu().numpy()
            #     hotword_scorer = HotwordScorer.build_scorer(None, weight=DEFAULT_HOTWORD_WEIGHT)
            #     PROCESSOR_WITH_LM.decoder._check_logits_dimension(logits)

            #     beam_search_outputs = decode_beams_ctc(
            #         PROCESSOR_WITH_LM.decoder,
            #         logits=logits,
            #         beam_width=args.beam_width,
            #         beam_prune_logp=DEFAULT_PRUNE_LOGP,
            #         token_min_logp=DEFAULT_MIN_TOKEN_LOGP,
            #         prune_history=DEFAULT_PRUNE_BEAMS,
            #         hotword_scorer=hotword_scorer,
            #         lm_start_state=None,
            #     )
            #     loss_weights = torch.softmax(torch.tensor([beam_search_output[-2] for beam_search_output in beam_search_outputs]), dim=-1)

            #     loss = 0
            #     for out_idx, beam_search_output in enumerate(beam_search_outputs):
            #         char_history = torch.tensor([*beam_search_output[-1]]).to(args.device)
            #         loss += loss_weights[out_idx] * criterion(outputs.squeeze(0) / args.temp, char_history)
            #     (loss / len(wavs)).backward(retain_graph=True)
    if 'beam_search_negative_sampling' in args.method:
        if args.not_blank:
            criterion = nn.CrossEntropyLoss(ignore_index=0)
        else:
            criterion = nn.CrossEntropyLoss()

        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)
            outputs = forward_batch(args, model, wav, torch.FloatTensor([lens[i]]).to(wav.device))
            logits = outputs.squeeze(0).detach().cpu().numpy()
            hotword_scorer = HotwordScorer.build_scorer(None, weight=DEFAULT_HOTWORD_WEIGHT)
            PROCESSOR_WITH_LM.decoder._check_logits_dimension(logits)

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
            char_history = torch.tensor([*beam_search_outputs[0][-1]]).to(args.device)

            if args.certain_only:
                selected_frame = []
                for frame_idx, (output, char_idx) in enumerate(zip(outputs.squeeze(0), char_history)):
                    probs = torch.softmax(output, dim=-1)
                    if probs[char_idx] > args.prob_threshold:
                        selected_frame.append(frame_idx)

                positive_outputs, positive_char_history = outputs.squeeze(0)[selected_frame].unsqueeze(0), char_history[selected_frame]
            else:
                positive_outputs, positive_char_history = outputs, char_history

            positive_loss = criterion(positive_outputs.squeeze(0) / args.temp, positive_char_history)
            (positive_loss / len(wavs)).backward(retain_graph=True)

            negative_loss = 0
            if args.negative_sampling_method == "random":
                for _ in range(args.num_negatives):
                    negative_char_history = torch.randint(high=outputs.shape[-1], size=(len(beam_search_outputs[0][-1]), )).to(args.device)
                    negative_mask = (negative_char_history != char_history) & (char_history != 0)

                    negative_outputs = []
                    for output, mask in zip(outputs.squeeze(0), negative_mask):
                        if mask:
                            negative_outputs.append(output)

                    if len(negative_outputs) > 0:
                        negative_outputs = torch.stack(negative_outputs).unsqueeze(0)
                        negative_char_history = torch.masked_select(negative_char_history, negative_mask)
                        negative_loss += -criterion(negative_outputs.squeeze(0) / args.temp, negative_char_history) / args.num_negatives
            elif args.negative_sampling_method == "beam_candidate":
                for out_idx in range(len(beam_search_outputs))[-args.num_negatives:]:
                    negative_char_history = torch.tensor([*beam_search_outputs[out_idx][-1]]).to(args.device)
                    negative_mask = (negative_char_history != char_history) & (char_history != 0)
                    negative_outputs = []
                    for output, mask in zip(outputs.squeeze(0), negative_mask):
                        if mask:
                            negative_outputs.append(output)

                    if len(negative_outputs) > 0:
                        negative_outputs = torch.stack(negative_outputs).unsqueeze(0)
                        negative_char_history = torch.masked_select(negative_char_history, negative_mask)

                        negative_loss += -criterion(negative_outputs.squeeze(0) / args.temp, negative_char_history) * (len(negative_char_history) / max(1, len(positive_char_history)))
            if torch.is_tensor(negative_loss):
                (args.ns_coef * negative_loss / len(wavs)).backward(retain_graph=True)
    if 'beam_em_mix' in args.method:
        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)
            outputs = forward_batch(args, model, wav, torch.FloatTensor([lens[i]]).to(wav.device))

            criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
            logits = outputs.squeeze(0).detach().cpu().numpy()
            hotword_scorer = HotwordScorer.build_scorer(None, weight=DEFAULT_HOTWORD_WEIGHT)

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
            char_history = torch.tensor([*beam_search_output[0][-1]]).to(args.device)
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
            outputs = forward_batch(args, model, wav, torch.FloatTensor([lens[i]]).to(wav.device))
            predicted_ids = torch.argmax(outputs, dim=-1)
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
            probs = torch.softmax(outputs[non_blank] / args.temp, dim=-1)
            mean_prob = torch.mean(probs, dim=0)
            loss = torch.sum(mean_prob * torch.log(mean_prob))
            (args.dm_coef * loss / len(wavs)).backward(retain_graph=True)

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
    processor = Wav2Vec2Processor.from_pretrained(args.asr, sampling_rate=sample_rate, return_attention_mask=True) if isinstance(model, Wav2Vec2ForCTC) else None

    if episodic:
        original_model_state, original_optimizer_state, original_scheduler_state = copy_model_and_optimizer(model, optimizer, scheduler)

    for batch_idx, batch in enumerate(dataset):
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
                    logits = forward_batch(args, model, wav, torch.FloatTensor([lens[i]]).to(wav.device))
                    probs.append(torch.softmax(logits.squeeze(0), dim=-1))
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
            model, optimizer, scheduler = load_model_and_optimizer(model, optimizer, scheduler, original_model_state, original_optimizer_state, original_scheduler_state)

        for step_idx in range(1, steps + 1):
            if adapt_or_not:
                forward_and_adapt(args, model, processor, optimizer, scheduler, wavs_to_adapt, lens_to_adapt)

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
                    memory_queue.popleft()[0]
                memory_queue.append((wav.cpu().detach(), prob.cpu().detach(), str(hash(wav.cpu().detach()))))

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