import os
import random
import gc
import logging
from datetime import datetime
from copy import deepcopy
# from queue import Queue

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.lobes.augment import TimeDomainSpecAugment, EnvCorrupt
from speechbrain.decoders.seq2seq import S2SRNNGreedySearcher, batch_filter_seq2seq_output
import speechbrain

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.common.parts.rnn import label_collate
from audio_augmentations import *

from jiwer import wer
import hydra
from omegaconf import OmegaConf

from data import load_dataset


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
        model = EncoderDecoderASR.from_hparams(args.asr, run_opts={"device": torch.device(args.device)}).requires_grad_(True).eval()
    elif args.asr == "pretrained_models/stt_en_conformer_transducer_small.nemo":
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(args.asr).to(torch.device(args.device)).requires_grad_(True).eval()
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
        # PolarityInversion()
        Noise(min_snr=0.01, max_snr=0.05),
        Gain(),
        Reverb(sample_rate=16000),
        HighLowPass(sample_rate=16000),
    ]
    strong_augmentation_list = [
        # Noise(min_snr=0.5, max_snr=1),
        # PitchShift(n_samples=16000*5, sample_rate=16000, pitch_cents_min=-1400, pitch_cents_max=1400),
        TimeDomainSpecAugment(
            perturb_prob=0, drop_freq_prob=1, drop_chunk_prob=1, speeds=[95, 100, 105],
            drop_freq_count_low=5, drop_freq_count_high=10, drop_chunk_count_low=5, drop_chunk_count_high=10,
            drop_chunk_length_low=500, drop_chunk_length_high=1000
        ),
    ]
    return weak_augmentation_list, strong_augmentation_list


def apply_augmentation(args, augmentation_list, wavs):
    augmentation = np.random.choice(augmentation_list)
    if isinstance(augmentation, TimeDomainSpecAugment):
        aug_wavs = augmentation(wavs, torch.ones(len(wavs)).to(wavs.device))
    else:
        aug_wavs = augmentation(wavs.cpu())
    return aug_wavs
        

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
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
        return model, optimizer, scheduler
    else: 
        return model, optimizer, None


def transcribe_batch(args, model, processor, wavs, lens):
    with torch.no_grad():
        if isinstance(model, Wav2Vec2ForCTC):
            inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding="longest")
            input_values = inputs.input_values.to(torch.device(args.device))
            outputs = model(input_values).logits
            predicted_ids = torch.argmax(outputs, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
        elif isinstance(model, EncoderDecoderASR):
            transcription, _ = model.transcribe_batch(wavs, wav_lens=torch.ones(len(wavs)).to(torch.device(args.device)))
        elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel): # conformer from nemo
            encoded_feature, encoded_len = model(input_signal=wavs, input_signal_length=lens)
            best_hyp_texts, _ = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded_feature, encoded_lengths=encoded_len, return_hypotheses=False
            )
            transcription = [best_hyp_text.upper() for best_hyp_text in best_hyp_texts]
    return transcription


def softmax_entropy(x, dim=-1):
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def mcc_loss(x, reweight=False, dim=-1, class_num=32):
    p = x.softmax(dim) # (1, L, D)
    p = p.squeeze(0) # (L, D)

    if reweight: # (1, L, D) * (L, 1) 
        target_entropy_weight = softmax_entropy(x, dim=-1).detach().squeeze(0) # instance-wise entropy (1, L, D)
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight) # (1, L)
        target_entropy_weight = x.shape[1] * target_entropy_weight / torch.sum(target_entropy_weight)
        cov_matrix_t = p.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(p)
    else:
        cov_matrix_t = p.transpose(1, 0).mm(p) # (D, L) * (L, D) -> (D, D)

    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
    return mcc_loss


def forward_and_adapt_ctc(args, model, teacher_model, processor, optimizer, scheduler, wavs, lens):
    inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to(torch.device(args.device))
    outputs = model(input_values).logits

    predicted_ids = torch.argmax(outputs, dim=-1)
    non_blank = torch.where(predicted_ids != 0, 1, 0).bool()

    loss = 0
    if args.method in ["original", "em_uncertainty", "em_sparse"]:
        if args.em_coef > 0:
            if args.method == "original":
                if args.not_blank:
                    e_loss = softmax_entropy(outputs / args.temp)[non_blank].mean(0).mean()
                else: 
                    e_loss = softmax_entropy(outputs / args.temp).mean(0).mean() 
            elif args.method == "em_uncertainty":
                if args.not_blank:
                    frame_weight = F.normalize(torch.reciprocal(softmax_entropy(outputs)[non_blank]), p=1, dim=-1).detach()
                    e_loss = torch.sum(frame_weight * softmax_entropy(outputs / args.temp)[non_blank], dim=-1).mean()
                else:
                    frame_weight = F.normalize(torch.reciprocal(softmax_entropy(outputs)), dim=-1).detach()
                    e_loss = torch.sum(frame_weight * softmax_entropy(outputs / args.temp), dim=-1).mean()
            elif args.method == "em_sparse":
                if args.not_blank:
                    selected_frame = non_blank & torch.where(softmax_entropy(outputs, dim=-1) < args.entropy_threshold, 1, 0).bool()
                    e_loss = softmax_entropy(outputs / args.temp)[selected_frame].mean(0).mean()
                else:
                    selected_frame = torch.where(softmax_entropy(outputs, dim=-1) < args.entropy_threshold, 1, 0).bool()
                    e_loss = softmax_entropy(outputs / args.temp)[selected_frame].mean(0).mean() 

            loss += args.em_coef * e_loss

        if 1 - args.em_coef > 0:
            c_loss = mcc_loss(outputs / args.temp, args.reweight)
            loss += (1 - args.em_coef) * c_loss
    elif args.method == "cr":
        weak_augmentation_list, strong_augmentation_list = get_augmentation(args)

        ce_loss = nn.CrossEntropyLoss()
        num_chunks = 4
        for sub_wav in input_values.chunk(num_chunks, dim=-1):
            weak_sub_wav = apply_augmentation(args, weak_augmentation_list, sub_wav).to(torch.device(args.device))

            with torch.no_grad():
                if teacher_model:
                    weak_outputs = teacher_model(weak_sub_wav).logits
                else:
                    weak_outputs = model(weak_sub_wav).logits

            weak_probs = F.softmax(weak_outputs, dim=-1)
            confidence, _ = torch.max(weak_probs, dim=-1, keepdim=True)
            weak_max_idx = torch.argmax(weak_probs, dim=-1, keepdim=True)
            weak_one_hots = torch.FloatTensor(weak_probs.shape).zero_().to(torch.device(args.device)).scatter(2, weak_max_idx, 1)
            non_blank = torch.where(weak_max_idx != 0, 1, 0).bool()

            selected_frames = non_blank & torch.where(confidence > args.prob_threshold, 1, 0).bool()
            selected_frames = selected_frames.expand_as(weak_probs)

            strong_sub_wav = apply_augmentation(args, strong_augmentation_list, sub_wav).to(torch.device(args.device))
            strong_outputs = model(strong_sub_wav).logits

            for strong_output, weak_one_hot, selected_frame in zip(strong_outputs, weak_one_hots, selected_frames): # element-wise loss in batch
                loss += ce_loss(
                    strong_output * selected_frame,
                    (weak_one_hot * selected_frame).detach()
                )
            del sub_wav, weak_sub_wav, weak_probs, confidence, weak_max_idx, non_blank, selected_frames, strong_sub_wav, strong_outputs

    optimizer.zero_grad()
    if not isinstance(loss, int):
        loss.backward()
    optimizer.step()
    if scheduler is not None: 
        scheduler.step()

    # check correlation
    inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to(torch.device(args.device))
    outputs = model(input_values).logits
    print(f"torch.max(softmax_entropy(outputs), dim=-1).values.shape : {torch.max(softmax_entropy(outputs), dim=-1).values.shape}")
    return torch.max(softmax_entropy(outputs), dim=-1).values.mean()


def forward_and_adapt_attn(args, model, teacher_model, processor, optimizer, scheduler, wavs, lens):
    def forward_attn(args, model, greedy_searcher, wavs, gt_wavs=None):
        log_probs_lst = []

        enc_states = model.encode_batch(wavs, wav_lens=torch.ones(len(wavs)).to(torch.device(args.device)))
        enc_lens = torch.tensor([enc_states.shape[1]]).to(torch.device(args.device))

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
                gt_enc_states = model.encode_batch(gt_wavs, wav_lens=torch.ones(len(gt_wavs)).to(torch.device(args.device)))
                gt_enc_lens = torch.tensor([gt_enc_states.shape[1]]).to(torch.device(args.device))

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

    greedy_searcher = S2SRNNGreedySearcher(
        model.mods.decoder.emb,
        model.mods.decoder.dec,
        model.mods.decoder.fc,
        **{
            "bos_index": model.mods.decoder.bos_index,
            "eos_index": model.mods.decoder.eos_index,
            "min_decode_ratio": model.mods.decoder.min_decode_ratio,
            "max_decode_ratio": model.mods.decoder.max_decode_ratio,
        },
    ).to(torch.device(args.device)).train()

    loss = 0
    if args.method in ["original", "em_uncertainty", "em_sparse"]:
        log_probs_lst = forward_attn(args, model, greedy_searcher, wavs)
        log_prob_tensor = torch.stack(log_probs_lst, dim=1)
        if args.em_coef > 0:
            if args.method == "original":
                e_loss = softmax_entropy(log_prob_tensor / args.temp, dim=-1).mean(0).mean()
            elif args.method == "em_uncertainty":
                frame_weight = F.normalize(torch.reciprocal(softmax_entropy(log_prob_tensor)), p=1, dim=-1).detach()
                e_loss = torch.sum(frame_weight * softmax_entropy(log_prob_tensor / args.temp), dim=-1).mean()
            elif args.method == "em_sparse":
                selected_frame = torch.where(softmax_entropy(log_prob_tensor, dim=-1) < args.entropy_threshold, 1, 0).bool()
                e_loss = softmax_entropy(log_prob_tensor / args.temp)[selected_frame].mean(0).mean()
            loss += args.em_coef * e_loss

        if 1 - args.em_coef > 0:
            c_loss = mcc_loss(log_prob_tensor / args.temp, reweight=args.reweight, class_num=1000)
            loss += (1 - args.em_coef) * c_loss
    if args.method == "cr":
        weak_augmentation_list, strong_augmentation_list = get_augmentation(args)
        # ce_loss = nn.CrossEntropyLoss()
        seq_loss = lambda x, y, z: speechbrain.nnet.losses.nll_loss(x, y, z, label_smoothing = 0.1)

        # outputs = torch.stack(forward_attn(args, model, greedy_searcher, wavs), dim=1)

        for _ in range(1):
            weak_wavs = apply_augmentation(args, weak_augmentation_list, wavs).to(torch.device(args.device))
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
                    ).to(torch.device(args.device)).train()
                    weak_outputs = torch.stack(forward_attn(args, teacher_model, teacher_greedy_searcher, weak_wavs), dim=1)
                else:
                    weak_outputs = torch.stack(forward_attn(args, model, greedy_searcher, weak_wavs), dim=1)

            weak_probs = F.softmax(weak_outputs, dim=-1)
            confidence, _ = torch.max(weak_probs, dim=-1, keepdim=True)
            weak_max_idx = torch.argmax(weak_probs, dim=-1, keepdim=True)
            # weak_one_hots = torch.FloatTensor(weak_probs.shape).zero_().to(torch.device(args.device)).scatter(2, weak_max_idx, 1)
            non_blank = torch.where(weak_max_idx != 0, 1, 0).bool()

            selected_frames = non_blank & torch.where(confidence > args.prob_threshold, 1, 0).bool()
            selected_frames = selected_frames.expand_as(weak_probs)

            # strong_wavs = strong_augmentation(wavs, torch.ones(len(wavs)).to(torch.device(args.device)))
            strong_wavs = apply_augmentation(args, strong_augmentation_list, wavs).to(torch.device(args.device))
            strong_outputs = torch.stack(forward_attn(args, model, greedy_searcher, strong_wavs, gt_wavs=weak_wavs), dim=1)

            loss += seq_loss(
                strong_outputs, torch.argmax(weak_probs, dim=-1).detach(), torch.ones(len(strong_outputs)).to(torch.device(args.device))
            )

            # for strong_output, weak_one_hot, selected_frame in zip(strong_outputs, weak_one_hots, selected_frames): # element-wise loss in batch
            #     loss += ce_loss(
            #         strong_output * selected_frame,
            #         (weak_one_hot * selected_frame).detach()
            #     )

            # def text_pipeline(words):
            #     tokens_list = model.tokenizer.encode_as_ids(words)
            #     tokens_bos = torch.LongTensor([model.hparams.bos_index] + (tokens_list))
            #     tokens_eos = torch.LongTensor(tokens_list + [model.hparams.eos_index]) # we use same eos and bos indexes as in pretrained model
            #     tokens = torch.LongTensor(tokens_list)
            #     return tokens_bos, tokens_eos

            # loss += seq_loss(
            #     outputs, torch.argmax(weak_probs, dim=-1).detach(), torch.ones(len(outputs)).to(torch.device(args.device))
            # )
        
        # del weak_wavs, weak_probs, weak_one_hots, confidence, non_blank, strong_wavs

    optimizer.zero_grad()
    if not isinstance(loss, int):
        loss.backward()
    optimizer.step()
    if scheduler is not None: 
        scheduler.step()

    log_probs_lst = forward_attn(args, model, greedy_searcher, wavs)
    log_prob_tensor = torch.stack(log_probs_lst, dim=1)
    return torch.max(softmax_entropy(log_prob_tensor), dim=-1).values.mean()


def forward_and_adapt_trans(args, model, teacher_model, processor, optimizer, scheduler, wavs, lens):
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

    loss = 0
    if args.method in ["original", "em_uncertainty", "em_sparse"]:
        log_probs_lst = forward_trans(args, model, wavs, lens, gt_wavs=None)
        log_prob_tensor = torch.stack(log_probs_lst, dim=1)
        if args.em_coef > 0:
            if args.method == "original":
                e_loss = softmax_entropy(log_prob_tensor / args.temp, dim=-1).mean(0).mean()
            elif args.method == "em_uncertainty":
                frame_weight = F.normalize(torch.reciprocal(softmax_entropy(log_prob_tensor)), p=1, dim=-1).detach()
                e_loss = torch.sum(frame_weight * softmax_entropy(log_prob_tensor / args.temp), dim=-1).mean()
            elif args.method == "em_sparse":
                selected_frame = torch.where(softmax_entropy(log_prob_tensor, dim=-1) < args.entropy_threshold, 1, 0).bool()
                e_loss = softmax_entropy(log_prob_tensor / args.temp)[selected_frame].mean(0).mean()
            loss += args.em_coef * e_loss

        if 1 - args.em_coef > 0:
            c_loss = mcc_loss(log_prob_tensor / args.temp, reweight=args.reweight, class_num=1000)
            loss += (1 - args.em_coef) * c_loss
    elif args.method == "cr":
        weak_augmentation_list, strong_augmentation_list = get_augmentation(args)

        ctc_loss = CTCLoss(num_classes=1000)

        # ce_loss = nn.CrossEntropyLoss()
        weak_wavs = apply_augmentation(args, weak_augmentation_list, wavs).to(torch.device(args.device))
        with torch.no_grad():
            if teacher_model:
                weak_outputs = torch.stack(forward_trans(args, teacher_model, weak_wavs, lens, gt_wavs=None), dim=1)
            else:
                weak_outputs = torch.stack(forward_trans(args, model, weak_wavs, lens, gt_wavs=None), dim=1)

        weak_probs = F.softmax(weak_outputs, dim=-1)
        confidence, _ = torch.max(weak_probs, dim=-1, keepdim=True)

        weak_max_idx = torch.argmax(weak_probs, dim=-1, keepdim=True)
        weak_one_hots = torch.FloatTensor(weak_probs.shape).zero_().to(torch.device(args.device)).scatter(2, weak_max_idx, 1)
        non_blank = torch.where(weak_max_idx != model.decoding.decoding._blank_index, 1, 0).bool()

        selected_frames = non_blank & torch.where(confidence > args.prob_threshold, 1, 0).bool()
        selected_frames = selected_frames.expand_as(weak_probs)

        del weak_outputs, weak_probs, confidence, non_blank

        strong_wavs = apply_augmentation(args, strong_augmentation_list, wavs).to(torch.device(args.device))
        strong_outputs = torch.stack(forward_trans(args, model, strong_wavs, lens, gt_wavs=weak_wavs), dim=1)
        # strong_outputs = torch.stack(forward_trans(args, model, strong_wavs, lens), dim=1)

        if strong_outputs.shape[1] > weak_one_hots.shape[1]:
            strong_outputs = strong_outputs[:, :weak_one_hots.shape[1], :]
        else:
            weak_one_hots = weak_one_hots[:, :strong_outputs.shape[1], :]
            selected_frames = selected_frames[:, :strong_outputs.shape[1], :]

        loss += ctc_loss(
            log_probs=strong_outputs,
            targets=weak_max_idx.squeeze(-1),
            input_lengths=torch.ones(len(strong_outputs)).to(torch.device(args.device)),
            target_lengths=torch.ones(len(strong_outputs)).to(torch.device(args.device))
        )

        # for strong_output, weak_one_hot, selected_frame in zip(strong_outputs, weak_one_hots, selected_frames): # element-wise loss in batch
        #     loss += ce_loss(
        #         strong_output * selected_frame,
        #         (weak_one_hot * selected_frame).detach()
        #     )
        del strong_wavs, strong_outputs

    optimizer.zero_grad()
    if not isinstance(loss, int):
        loss.backward()
    optimizer.step()
    if scheduler is not None: 
        scheduler.step()

    # check correlation
    log_probs_lst = forward_trans(args, model, wavs, lens, gt_wavs=None)
    log_prob_tensor = torch.stack(log_probs_lst, dim=1)
    return torch.max(softmax_entropy(log_prob_tensor), dim=-1).values.mean()


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

    # TODO: implement memory queue
    # use_memory_queue = args.use_memory_queue
    # queue_size = args.queue_size
    # n_neighbors = args.n_neighbors
    # if use_memory_queue:
    #     memory_queue = Queue(maxsize=queue_size)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = get_logger(args)

    with open(os.path.join(args.log_dir, "config.yaml"), 'w') as f:
        OmegaConf.save(args, f)
    logger.info(OmegaConf.to_yaml(args))

    dataset = load_dataset(split, dataset_name, dataset_dir, batch_size, extra_noise)
    gt_texts, ori_transcriptions, transcriptions_1, transcriptions_3, transcriptions_5, transcriptions_10, transcriptions_20, transcriptions_40 = [], [], [], [], [], [], [], []

    # for chceking
    entropy_list, wer_list = [], []

    original_model = get_model(args, original=True)
    model = get_model(args, original=False)
    if isinstance(model, Wav2Vec2ForCTC): # ctc
        params, _ = collect_params_ctc(model, train_params, bias_only)
    elif isinstance(model, EncoderDecoderASR):
        params, _ = collect_params_attn(model, train_params, bias_only)
    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
        params, _ = collect_params_trans(model, train_params, bias_only)
    optimizer, scheduler = get_optimizer(params, optimizer, lr, scheduler=scheduler)

    teacher_model = get_model(args, original=False) if teacher_student else None
    processor = Wav2Vec2Processor.from_pretrained(args.asr, sampling_rate=sample_rate, return_attention_mask=True) if isinstance(model, Wav2Vec2ForCTC) else None

    if episodic:
        original_model_state, original_optimizer_state, original_scheduler_state = copy_model_and_optimizer(model, optimizer, scheduler)

    for batch_idx, batch in enumerate(dataset):
        lens, wavs, texts, _ = batch
        if not isinstance(model, Wav2Vec2ForCTC):
            wavs = torch.tensor(np.array(wavs)).to(torch.device(args.device))
        lens = lens.to(torch.device(args.device))
        gt_texts += texts

        ori_transcription = transcribe_batch(args, original_model, processor, wavs, lens)
        ori_transcriptions += ori_transcription
        ori_wer = wer(list(texts), list(ori_transcription))
        logger.info(f"{batch_idx}/{len(dataset)}")
        logger.info(f"original WER: {ori_wer}")

        if episodic:
            model, optimizer, scheduler = load_model_and_optimizer(model, optimizer, scheduler, original_model_state, original_optimizer_state, original_scheduler_state)

        logger.info(f"gt : {list(texts)}")
        logger.info(f"original transcrption : {list(ori_transcription)}")

        for step_idx in range(1, steps + 1):
            if isinstance(model, Wav2Vec2ForCTC): # ctc
                entropy = forward_and_adapt_ctc(args, model, teacher_model, processor, optimizer, scheduler, wavs, lens)
            elif isinstance(model, EncoderDecoderASR): # attention-based encoder-decoder
                model.train()
                entropy = forward_and_adapt_attn(args, model, teacher_model, processor, optimizer, scheduler, wavs, lens)
                model.eval()
            elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel): # transducer
                model.train()
                entropy = forward_and_adapt_trans(args, model, teacher_model, processor, optimizer, scheduler, wavs, lens)
                model.eval()

            if step_idx in [1, 3, 5, 10, 20, 40]:
                transcription = transcribe_batch(args, model, processor, wavs, lens)
                transcription_list = eval(f"transcriptions_{step_idx}")
                transcription_list += transcription

                ada_wer = wer(list(texts), list(transcription))
                logger.info(f"adapt-{step_idx} WER: {ada_wer}")
                logger.info(f"{' '.join(list(texts))}, {' '.join(list(transcription))}")

                entropy_list.append(entropy.detach().cpu().numpy())
                wer_list.append(ada_wer)

        if stochastic_restoration:
            for model_param, original_param in zip(model.parameters(), original_model.parameters()):
                restore = np.random.binomial(n=1, p=restore_prob, size=1)[0]
                with torch.no_grad():
                    model_param.copy_((1 - restore) * model_param + restore * original_param)

        if teacher_student:
            for teacher_param, model_param in zip(teacher_model.parameters(), model.parameters()):
                with torch.no_grad():
                    teacher_param.copy_(momentum * teacher_param + (1 - momentum) * model_param)

        gc.collect()
        torch.cuda.empty_cache()
        logger.info("\n\n\n\n\n\n")

        import matplotlib.pyplot as plt
        plt.scatter(entropy_list, wer_list)
        plt.xlabel('entropy')
        plt.ylabel('wer')
        plt.savefig("entropy_wer.png")


    logger.info(f"number of data : {len(dataset)}")
    logger.info(f"original WER: {wer(gt_texts, ori_transcriptions)}")
    for step_idx in [1, 3, 5, 10, 20, 40]:
        if step_idx <= steps:
            transcription_list = eval(f"transcriptions_{step_idx}")
            logger.info(f"TTA-{step_idx}: {wer(gt_texts, transcription_list)}")



if __name__ == '__main__':
    main()