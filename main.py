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

import hydra
from omegaconf import OmegaConf, open_dict

import numpy as np
from sklearn.decomposition import PCA
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from info_nce import InfoNCE

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import speechbrain
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.lobes.augment import TimeDomainSpecAugment
from speechbrain.decoders.seq2seq import S2SRNNGreedySearcher
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.common.parts.rnn import label_collate

from audio_augmentations import *
import sentencepiece
from jiwer import wer

from data import load_dataset
from forward import *


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


def transcribe_batch(args, model, processor, wavs, lens):
    with torch.no_grad():
        if isinstance(model, Wav2Vec2ForCTC):
            outputs = model(wavs).logits
            predicted_ids = torch.argmax(outputs, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
        elif isinstance(model, EncoderDecoderASR):
            transcription = []
            for wav in wavs:
                wav = wav.unsqueeze(0)
                text = model.transcribe_batch(wav, wav_lens=torch.ones(len(wav)).to(args.device))[0]
                transcription.append(text[0])
        elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel): # conformer from nemo
            transcription = []
            for wav in wavs:
                wav = wav.unsqueeze(0)
                encoded_feature, encoded_len = model(input_signal=wav, input_signal_length=lens)
                best_hyp_texts, _ = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=encoded_feature, encoded_lengths=encoded_len, return_hypotheses=False
                )
                text = [best_hyp_text.upper() for best_hyp_text in best_hyp_texts][0]
                transcription.append(text)
    return transcription


def softmax_entropy(x, dim=-1):
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def non_saturating_loss(x, dim=-1):
    max_idx = torch.argmax(x, dim=dim, keepdim=True)
    one_hots = torch.zeros_like(x).scatter(-1, max_idx, 1).to(x.device)
    return torch.mean(one_hots * x) + torch.log(((1 - one_hots) * torch.exp(x)).sum(dim=dim)).mean()


def mcc_loss(x, reweight=False, dim=-1, class_num=32):
    mcc_loss = 0
    for x_split in x: # (B, L, D) -> (L, D)
        x_split = x_split.unsqueeze(0)
        p = x_split.softmax(dim) # (1, L, D)
        p = p.squeeze(0) # (L, D)

        if reweight: # (1, L, D) * (L, 1)
            target_entropy_weight = softmax_entropy(x_split, dim=-1).detach().squeeze(0) # instance-wise entropy (1, L, D)
            target_entropy_weight = 1 + torch.exp(-target_entropy_weight) # (1, L)
            target_entropy_weight = x_split.shape[1] * target_entropy_weight / torch.sum(target_entropy_weight)
            cov_matrix_t = p.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(p)
        else:
            cov_matrix_t = p.transpose(1, 0).mm(p) # (D, L) * (L, D) -> (D, D)

        cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
        mcc_loss += (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
    mcc_loss /= len(x)
    return mcc_loss


def js_divergence(p1, p2):
    total_m = 0.5 * (p1 + p2)
    loss = 0.5 * F.kl_div(torch.log(p1), total_m, reduction="batchmean") + 0.5 * F.kl_div(torch.log(p2), total_m, reduction="batchmean")
    return loss


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
    out_wavs = [wav for wav in wavs]
    if len(memory_queue) <= args.n_neighbors:
        for wav, _ in list(memory_queue):
            out_wavs.append(wav.to(wavs.device))
    elif method == "random":
        selected_instances = random.sample(list(memory_queue), args.n_neighbors)
        for wav, _ in selected_instances:
            out_wavs.append(wav.to(wavs.device))
    elif method == "latest":
        selected_instances = list(memory_queue)[-args.n_neighbors:]
        for wav, _ in selected_instances:
            out_wavs.append(wav.to(wavs.device))
    elif method == "informative":
        new_mean_probs = torch.mean(probs.view(-1, probs.shape[-1]), dim=0)
        previous_instances = list(memory_queue)
        selected_instances = []

        for wav, prob in previous_instances:
            entropy = torch.mean(- torch.sum(prob * torch.log(prob), dim=-1)) # mean over tokens
            mean_probs = torch.mean(prob, dim=0).to(wavs.device)
            js_div = js_divergence(new_mean_probs, mean_probs)
            selected_instances.append((wav, entropy / js_div))

        for wav, _ in sorted(selected_instances, key=lambda x: x[1], reverse=True)[:args.n_neighbors]:
            out_wavs.append(wav.to(wavs.device))
    out_wavs = pad_sequence(out_wavs, batch_first=True)
    return out_wavs


def forward_and_adapt_ctc(args, model, teacher_model, processor, optimizer, scheduler, wavs, lens):
    outputs = model(wavs).logits
    predicted_ids = torch.argmax(outputs, dim=-1)
    non_blank = torch.where(predicted_ids != 0, 1, 0).bool()

    optimizer.zero_grad()
    if "original" in args.method or "em_uncertainty" in args.method or "em_sparse" in args.method:
        if args.em_coef > 0:
            if "original" in args.method:
                if args.not_blank:
                    e_loss = softmax_entropy(outputs / args.temp)[non_blank].mean(0).mean()
                else: 
                    e_loss = softmax_entropy(outputs / args.temp).mean(0).mean() 
            elif "em_uncertainty" in args.method:
                if args.not_blank:
                    frame_weight = F.normalize(torch.reciprocal(softmax_entropy(outputs)[non_blank]), p=1, dim=-1).detach()
                    e_loss = torch.sum(frame_weight * softmax_entropy(outputs / args.temp)[non_blank], dim=-1).mean()
                else:
                    frame_weight = F.normalize(torch.reciprocal(softmax_entropy(outputs)), dim=-1).detach()
                    e_loss = torch.sum(frame_weight * softmax_entropy(outputs / args.temp), dim=-1).mean()
            elif "em_sparse" in args.method:
                if args.not_blank:
                    selected_frame = non_blank & torch.where(softmax_entropy(outputs, dim=-1) < args.entropy_threshold, 1, 0).bool()
                    e_loss = softmax_entropy(outputs / args.temp)[selected_frame].mean(0).mean()
                else:
                    selected_frame = torch.where(softmax_entropy(outputs, dim=-1) < args.entropy_threshold, 1, 0).bool()
                    e_loss = softmax_entropy(outputs / args.temp)[selected_frame].mean(0).mean() 
            (args.em_coef * e_loss).backward(retain_graph=True)

        if 1 - args.em_coef > 0:
            c_loss = mcc_loss(outputs / args.temp, args.reweight)
            ((1 - args.em_coef) * c_loss).backward(retain_graph=True)
    if "cr" in args.method:
        weak_augmentation_list, strong_augmentation_list = get_augmentation(args)

        ce_loss = nn.CrossEntropyLoss()
        for sub_wav in input_values: # element-wise iteration
            sub_wav = sub_wav.unsqueeze(0)
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
                ) / (len(input_values) * len(strong_outputs))
                cr_loss.backward(retain_graph=True)
            del sub_wav, weak_sub_wav, weak_probs, confidence, weak_max_idx, non_blank, selected_frames, strong_sub_wav, strong_outputs

    optimizer.step()
    if scheduler is not None:
        scheduler.step()


def forward_and_adapt_attn(args, model, teacher_model, processor, optimizer, scheduler, wavs, lens, adapter=None, step_idx=None):
    greedy_searcher = S2SRNNGreedySearcher(model.mods.decoder.emb, model.mods.decoder.dec, model.mods.decoder.fc, **{"bos_index": model.mods.decoder.bos_index, "eos_index": model.mods.decoder.eos_index, "min_decode_ratio": model.mods.decoder.min_decode_ratio, "max_decode_ratio": model.mods.decoder.max_decode_ratio,},).to(args.device).train()
    optimizer.zero_grad()

    current = time.time()

    if "original" in args.method or "em_uncertainty" in args.method or "em_sparse" in args.method:
        for wav in wavs:
            wav = wav.unsqueeze(0)
            log_probs_lst = forward_attn(args, model, greedy_searcher, wav)

            print(f"7-1 : {time.time() - current}")
            current = time.time()

            log_prob_tensor = torch.stack(log_probs_lst, dim=1)
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
                (args.em_coef / len(wavs) * e_loss).backward()

                # print(f"(args.em_coef / len(wavs) * e_loss) : {(args.em_coef / len(wavs) * e_loss)}")
                # print(f"model.require_grad : {model.require_grad}")
                # grad_dic = {x[0]:x[1].data.grad for x in model.named_parameters()}
                # for k, v in grad_dic.items():
                #     print(f"{k} : {v}")
                # for x in mode.named_parameters():

            if 1 - args.em_coef > 0:
                c_loss = mcc_loss(log_prob_tensor / args.temp, reweight=args.reweight, class_num=1000)
                ((1 - args.em_coef) / len(wavs) * c_loss).backward()
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
                weak_outputs = torch.stack(forward_attn(args, teacher_model, teacher_greedy_searcher, weak_wavs), dim=1)
            else:
                weak_outputs = torch.stack(forward_attn(args, model, greedy_searcher, weak_wavs), dim=1)

        weak_probs = F.softmax(weak_outputs, dim=-1)
        confidence, _ = torch.max(weak_probs, dim=-1, keepdim=True)
        weak_max_idx = torch.argmax(weak_probs, dim=-1, keepdim=True)
        non_blank = torch.where(weak_max_idx != 0, 1, 0).bool()

        selected_frames = non_blank & torch.where(confidence > args.prob_threshold, 1, 0).bool()
        selected_frames = selected_frames.expand_as(weak_probs)

        strong_wavs = apply_augmentation(args, strong_augmentation_list, wavs).to(args.device)
        strong_outputs = torch.stack(forward_attn(args, model, greedy_searcher, strong_wavs, gt_wavs=weak_wavs), dim=1)

        cr_loss = seq_loss(
            strong_outputs, torch.argmax(weak_probs, dim=-1).detach(), torch.ones(len(strong_outputs)).to(args.device)
        )
        cr_loss.backward()
    if "cr_feature" in args.method:
        weak_augmentation_list, strong_augmentation_list = get_augmentation(args)

        for sub_wav in wavs:
            sub_wav = sub_wav.unsqueeze(0)
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
        for wav in wavs:
            wav = wav.unsqueeze(0)
            log_probs_lst = forward_attn(args, model, greedy_searcher, wav)
            log_prob_tensor = torch.stack(log_probs_lst, dim=1)
            max_log_probs, _ = torch.max(log_prob_tensor, dim=-1, keepdim=False)
            sum_log_probs = torch.sum(max_log_probs, dim=-1)
            nll_loss = - sum_log_probs.mean()
            (nll_loss / len(wavs)).backward()
    if "p_logp" in args.method:
        for wav in wavs:
            wav = wav.unsqueeze(0)
            log_probs_lst = forward_attn(args, model, greedy_searcher, wav)
            log_prob_tensor = torch.stack(log_probs_lst, dim=1)
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
 
        for wav in wavs:
            wav = wav.unsqueeze(0)
            for i in range(num_augs):
                if i > 0:
                    aug_wav = apply_augmentation(args, strong_augmentation_list, wav).to(args.device)
                else:
                    aug_wav = wav

                log_probs_lst = forward_attn(args, model, greedy_searcher, aug_wav)
                log_prob_tensor = torch.stack(log_probs_lst, dim=1)
                predicted_ids = torch.argmax(log_prob_tensor, dim=-1)

                mean_prob = torch.mean(torch.softmax(log_prob_tensor, dim=-1), dim=0, keepdim=True)
                mean_log_prob = torch.mean(torch.log_softmax(log_prob_tensor, dim=-1), dim=0, keepdim=True)

                e_loss = - torch.sum(mean_prob * mean_log_prob, dim=-1).mean()
                (e_loss / (len(wavs) * num_augs)).backward()

    print(f"7-3 : {time.time() - current}")
    current = time.time()

    optimizer.step()

    print(f"7-4 : {time.time() - current}")
    current = time.time()

    if scheduler is not None: 
        scheduler.step()


def forward_and_adapt_trans(args, model, teacher_model, processor, optimizer, scheduler, wavs, lens):
    optimizer.zero_grad()
    if "original" in args.method or "em_uncertainty" in args.method or "em_sparse" in args.method:
        for i, wav in enumerate(wavs):
            wav = wav.unsqueeze(0)
            log_probs_lst = forward_trans(args, model, wav, torch.tensor([lens[i]]).to(wav.device), gt_wavs=None)
            log_prob_tensor = torch.stack(log_probs_lst, dim=1)
            if args.em_coef > 0:
                if "original" in args.method:
                    e_loss = softmax_entropy(log_prob_tensor / args.temp, dim=-1).mean(0).mean()
                elif "em_uncertainty" in args.method:
                    frame_weight = F.normalize(torch.reciprocal(softmax_entropy(log_prob_tensor)), p=1, dim=-1).detach()
                    e_loss = torch.sum(frame_weight * softmax_entropy(log_prob_tensor / args.temp), dim=-1).mean()
                elif "em_sparse" in args.method:
                    selected_frame = torch.where(softmax_entropy(log_prob_tensor, dim=-1) < args.entropy_threshold, 1, 0).bool()
                    e_loss = softmax_entropy(log_prob_tensor / args.temp)[selected_frame].mean(0).mean()
                ((args.em_coef / len(wavs)) * e_loss).backward()

            if 1 - args.em_coef > 0:
                c_loss = mcc_loss(log_prob_tensor / args.temp, reweight=args.reweight, class_num=1000)
                (((1 - args.em_coef) / len(wavs)) * c_loss).backward()
    if "cr" in args.method:
        weak_augmentation_list, strong_augmentation_list = get_augmentation(args)
        ctc_loss = CTCLoss(num_classes=1000)

        weak_wavs = apply_augmentation(args, weak_augmentation_list, wavs).to(args.device)
        with torch.no_grad():
            if teacher_model:
                weak_outputs = torch.stack(forward_trans(args, teacher_model, weak_wavs, lens, gt_wavs=None), dim=1)
            else:
                weak_outputs = torch.stack(forward_trans(args, model, weak_wavs, lens, gt_wavs=None), dim=1)

        weak_probs = F.softmax(weak_outputs, dim=-1)
        confidence, _ = torch.max(weak_probs, dim=-1, keepdim=True)

        weak_max_idx = torch.argmax(weak_probs, dim=-1, keepdim=True)
        weak_one_hots = torch.FloatTensor(weak_probs.shape).zero_().to(args.device).scatter(2, weak_max_idx, 1)
        non_blank = torch.where(weak_max_idx != model.decoding.decoding._blank_index, 1, 0).bool()

        selected_frames = non_blank & torch.where(confidence > args.prob_threshold, 1, 0).bool()
        selected_frames = selected_frames.expand_as(weak_probs)

        del weak_outputs, weak_probs, confidence, non_blank

        strong_wavs = apply_augmentation(args, strong_augmentation_list, wavs).to(args.device)
        strong_outputs = torch.stack(forward_trans(args, model, strong_wavs, lens, gt_wavs=weak_wavs), dim=1)

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
    logger = get_logger(args)
    logger.info(OmegaConf.to_yaml(args))

    use_memory_queue = args.use_memory_queue
    queue_size = args.queue_size
    n_neighbors = args.n_neighbors
    if use_memory_queue:
        global memory_queue
        memory_queue = deque([], maxlen=queue_size)

    dataset = load_dataset(split, dataset_name, dataset_dir, batch_size, extra_noise)
    gt_texts, ori_transcriptions, transcriptions_1, transcriptions_3, transcriptions_5, transcriptions_10, transcriptions_20, transcriptions_40 = [], [], [], [], [], [], [], []

    original_model = get_model(args, original=True)
    model = get_model(args, original=False)

    current = time.time()

    if isinstance(model, Wav2Vec2ForCTC): # ctc
        params, _ = collect_params_ctc(model, train_params, bias_only)
    elif isinstance(model, EncoderDecoderASR):
        params, _ = collect_params_attn(model, train_params, bias_only)
    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
        params, _ = collect_params_trans(model, train_params, bias_only)
    optimizer, scheduler = get_optimizer(params, opt_name=args.optimizer, lr=lr, scheduler=args.scheduler)

    print(f"1 : {time.time() - current}")
    current = time.time()

    teacher_model = get_model(args, original=False) if teacher_student else None
    processor = Wav2Vec2Processor.from_pretrained(args.asr, sampling_rate=sample_rate, return_attention_mask=True) if isinstance(model, Wav2Vec2ForCTC) else None

    if episodic:
        original_model_state, original_optimizer_state, original_scheduler_state = copy_model_and_optimizer(model, optimizer, scheduler)

    print(f"2 : {time.time() - current}")
    current = time.time()

    # TODO: casuality
    ori_wer_list, ada_wer_list, diff_wer_list, max_ent_list, avg_ent_list, max_ood_list, avg_ood_list = [], [], [], [], [], [], []

    for batch_idx, batch in enumerate(dataset):
        lens, wavs, texts, _ = batch

        print(f"3 : {time.time() - current}")
        current = time.time()

        if isinstance(model, Wav2Vec2ForCTC):
            wavs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding="longest").input_values.to(args.device)
        else:
            wavs = pad_sequence([torch.from_numpy(wav) for wav in wavs], batch_first=True).to(args.device)
        lens = lens.to(args.device)

        print(f"4 : {time.time() - current}")
        current = time.time()

        if args.use_memory_queue:
            probs = []
            for i, wav in enumerate(wavs):
                wav = wav.unsqueeze(0)
                if isinstance(model, Wav2Vec2ForCTC):
                    logit = model(wav).logits
                elif isinstance(model, EncoderDecoderASR):
                    greedy_searcher = S2SRNNGreedySearcher(model.mods.decoder.emb, model.mods.decoder.dec, model.mods.decoder.fc, **{"bos_index": model.mods.decoder.bos_index, "eos_index": model.mods.decoder.eos_index, "min_decode_ratio": model.mods.decoder.min_decode_ratio, "max_decode_ratio": model.mods.decoder.max_decode_ratio,},).to(args.device)
                    logit = forward_attn(args, model, greedy_searcher, wav, gt_wavs=None)
                    logit = torch.stack(logit, dim=1)
                elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
                    logit = forward_trans(args, model, wav, torch.tensor([lens[i]]).to(wav.device), gt_wavs=None)
                    logit = torch.stack(logit, dim=1)
            probs.append(torch.softmax(logit.squeeze(0), dim=-1))
            probs = pad_sequence(probs, batch_first=True).to(args.device)
            
            wavs_to_adapt = get_instance_from_queue(args, args.queue_method, wavs, probs)
            lens_to_adapt = torch.tensor([len(wav) for wav in wavs_to_adapt]).to(args.device)
        else:
            wavs_to_adapt = wavs
            lens_to_adapt = lens

        print(f"5 : {time.time() - current}")
        current = time.time()

        gt_texts += texts
        ori_transcription = transcribe_batch(args, original_model, processor, wavs, lens)
        ori_transcriptions += ori_transcription
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

            print(f"6 : {time.time() - current}")
            current = time.time()

            if isinstance(model, Wav2Vec2ForCTC): # ctc
                forward_and_adapt_ctc(args, model, teacher_model, processor, optimizer, scheduler, wavs_to_adapt, lens_to_adapt)
            elif isinstance(model, EncoderDecoderASR): # attention-based encoder-decoder
                forward_and_adapt_attn(args, model, teacher_model, processor, optimizer, scheduler, wavs_to_adapt, lens_to_adapt, adapter=adapter, step_idx=step_idx)
            elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel): # transducer
                forward_and_adapt_trans(args, model, teacher_model, processor, optimizer, scheduler, wavs_to_adapt, lens_to_adapt)

            print(f"6-1 : {time.time() - current}")
            current = time.time()

            if step_idx in [1, 3, 5, 10, 20, 40]:
                transcription = transcribe_batch(args, model, processor, wavs, lens)
                transcription_list = eval(f"transcriptions_{step_idx}")
                transcription_list += transcription

                ada_wer = wer(list(texts), list(transcription))
                logger.info(f"adapt-{step_idx} WER: {ada_wer}")
                logger.info(f"adapt-{step_idx} text: {' '.join(list(transcription))}")

            print(f"6-2 : {time.time() - current}")
            current = time.time()

            gc.collect()
            torch.cuda.empty_cache()

        # TODO: casuality
        if args.use_memory_queue:
            probs = probs.view(-1, probs.shape[-1])

            per_token_entropy = - torch.sum(probs * torch.log(probs), dim=-1)
            max_entropy, _ = torch.max(per_token_entropy, dim=0)
            avg_entropy = torch.mean(per_token_entropy, dim=0)

            per_token_ood, _ = torch.max(probs, dim=-1)
            per_token_odd = - per_token_ood
            max_ood, _ = torch.max(per_token_odd, dim=0)
            avg_ood = torch.mean(per_token_odd, dim=0)

            ori_wer_list.append(ori_wer)
            ada_wer_list.append(ada_wer)
            diff_wer_list.append(ada_wer - ori_wer)
            max_ent_list.append(max_entropy.item())
            avg_ent_list.append(avg_entropy.item())
            max_ood_list.append(max_ood.item())
            avg_ood_list.append(avg_ood.item())

            import pandas as pd
            write_dict = {
                'ori_wer': ori_wer_list,
                'ada_wer': ada_wer_list,
                'diff_wer': diff_wer_list,
                'max_ent': max_ent_list,
                'avg_ent': avg_ent_list,
                'max_ood': max_ood_list,
                'avg_ood': avg_ood_list
            }
            df = pd.DataFrame(write_dict)
            df.to_csv(f"casuality_{args.dataset_name}_{list(args.method)}.csv", index=False)

        if args.use_memory_queue:
            for wav, prob in zip(wavs, probs):
                while len(memory_queue) >= args.queue_size:
                    memory_queue.popleft()
                memory_queue.append((wav.cpu().detach(), prob.cpu().detach()))

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
    main()