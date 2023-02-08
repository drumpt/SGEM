import time
import random
import os
import gc
import logging
from copy import deepcopy
from datetime import datetime
import hydra
from omegaconf import OmegaConf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
from torch.optim.lr_scheduler import CosineAnnealingLR
# from apex import amp
# from torch.cuda.amp import GradScaler
# from torch.cuda.amp import autocast

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from speechbrain.pretrained import EncoderDecoderASR
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules.rnnt_beam_decoding import BeamRNNTInfer
from jiwer import wer

from data import *
from forward import *
from loss import *


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_handler = logging.FileHandler(os.path.join(args.log_dir, f"log_{time_string}.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_model(args):
    if args.asr in ["facebook/wav2vec2-base-960h"]: # CTC-based models
        model = Wav2Vec2ForCTC.from_pretrained(args.asr).requires_grad_(True).eval()
        if 'cuda' in args.device:
            model = model.cuda()
    elif args.asr in ["speechbrain/asr-crdnn-rnnlm-librispeech", "speechbrain/asr-crdnn-transformerlm-librispeech", "speechbrain/asr-transformer-transformerlm-librispeech", "speechbrain/asr-conformersmall-transformerlm-librispeech"]: # attention-based models
        model = EncoderDecoderASR.from_hparams(args.asr, run_opts={"device": args.device}).requires_grad_(True).eval()
    elif args.asr in ["pretrained_models/stt_en_conformer_transducer_small.nemo", "pretrained_models/stt_en_conformer_transducer_large.nemo"]: # transducers
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(args.asr).to(args.device).requires_grad_(True).eval()
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


def freeze_norm_stats(model):
    for _, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            m.track_running_stats = False


def set_rnn_to_train(model):
    if isinstance(model, EncoderDecoderASR) or isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
        for nm, m in model.named_modules():
            if isinstance(m, torch.nn.modules.rnn.RNNBase):
                m.train()
                m.dropout = 0
    return model


def get_optimizer(args, params, opt_name='AdamW', lr=1e-4, beta=0.9, weight_decay=0., scheduler=None):
    opt = getattr(torch.optim, opt_name)
    if opt_name == 'Adam':       
        optimizer = opt(params, lr=lr, betas=(beta, 0.999), weight_decay=weight_decay)
    else: 
        optimizer = opt(params, lr=lr, weight_decay=weight_decay)
    
    if scheduler is not None: 
        return optimizer, eval(scheduler)(optimizer, T_max=args.t_max, eta_min=args.eta_min)
    return optimizer, None


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


def get_blank_index(args, model, processor):
    if isinstance(model, Wav2Vec2ForCTC):
        blank_index = 0
    elif isinstance(model, EncoderDecoderASR):
        blank_index = model.mods.decoder.blank_index
    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
        blank_index = processor.blank
    return blank_index


def forward_and_adapt(args, model, processor, optimizer, scheduler, wavs, lens):
    optimizer.zero_grad()
    blank_index = get_blank_index(args, model, decoder_processor)

    if "original" in args.method or "em_uncertainty" in args.method or "em_sparse" in args.method:
        for i, wav in enumerate(wavs):
            wav = wav.unsqueeze(0)[:lens[i]]
            outputs, _ = get_logits_and_pseudo_labels(args, model, decoder_processor, wav, torch.FloatTensor([lens[i]]).to(wav.device))
            predicted_ids = torch.argmax(outputs, dim=-1)
            non_blank = torch.where(predicted_ids != blank_index, 1, 0).bool()

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
                c_loss = mcc_loss(outputs / args.temp, class_num=outputs.shape[-1], reweight=True)
                ((1 - args.em_coef) * c_loss / (len(wavs))).backward(retain_graph=True)
    if "em_dropout" in args.method:
        for i, wav in enumerate(wavs):
            wav = wav.unsqueeze(0)[:lens[i]]
            e_loss = 0
            model.train()
            NUM_DROPOUTS = 5
            for _ in range(NUM_DROPOUTS):
                outputs, _ = get_logits_and_pseudo_labels(args, model, decoder_processor, wav, torch.FloatTensor([lens[i]]).to(wav.device))
                predicted_ids = torch.argmax(outputs, dim=-1)
                non_blank = torch.where(predicted_ids != blank_index, 1, 0).bool()
                e_loss += (softmax_entropy(outputs / args.temp)[non_blank].mean(0).mean()) / NUM_DROPOUTS
            model.eval()
            model = set_rnn_to_train(model)
            (args.em_coef * e_loss / (len(wavs))).backward(retain_graph=True)
    if "greedy_pseudo_labeling" in args.method:
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
                non_blank = torch.where(predicted_ids != blank_index, 1, 0).bool()
                max_log_probs = non_blank * max_log_probs

            sum_log_probs = torch.sum(max_log_probs, dim=-1)

            nll_loss = - sum_log_probs.mean()
            (nll_loss / len(wavs)).backward()

            del wav, log_prob_tensor, max_log_probs, sum_log_probs, nll_loss
    if "gce" in args.method:
        q = 0.7
        for i, wav in enumerate(wavs):
            wav = wav.unsqueeze(0)[:lens[i]]
            log_prob_tensor, _ = get_logits_and_pseudo_labels(args, model, decoder_processor, wav, torch.FloatTensor([lens[i]]).to(wav.device))
            probs = torch.softmax(log_prob_tensor / args.temp, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1, keepdim=False)

            if args.certain_only:
                confidence, _ = torch.max(torch.softmax(log_prob_tensor, dim=-1), dim=-1, keepdim=True)
                selected_tokens = torch.where(confidence > args.prob_threshold, 1, 0).squeeze(2).detach()
                max_probs = selected_tokens * max_probs

            if args.not_blank:
                predicted_ids = torch.argmax(probs, dim=-1)
                non_blank = torch.where(predicted_ids != blank_index, 1, 0).bool()
                max_probs = non_blank * max_probs

            gce_loss = torch.mean((1 - max_probs ** q) / q, dim=-1)
            (gce_loss / len(wavs)).backward()

            del wav, log_prob_tensor, max_probs, gce_loss
    if 'ctc' in args.method:
        for i, wav in enumerate(wavs):
            # TODO: implement this
            vocab = CTC_VOCAB

            wav = wav[:lens[i]].unsqueeze(0)
            outputs, _ = get_logits_and_pseudo_labels(args, model, decoder_processor, wav, torch.FloatTensor([lens[i]]).to(wav.device))
            
            ctc_loss = pl_loss(outputs, vocab, processor)
            (ctc_loss / len(wavs)).backward()
    if 'beam_search_max' in args.method or 'beam_search_all' in args.method or 'beam_search_negative_sampling' in args.method:
        for i, wav in enumerate(wavs):
            criterion = nn.CrossEntropyLoss(ignore_index=blank_index, label_smoothing=args.label_smoothing) if args.not_blank else nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

            wav = wav[:lens[i]].unsqueeze(0)

            import time
            current = time.time()

            outputs, pseudo_labels = get_logits_and_pseudo_labels(args, model, decoder_processor, wav, torch.FloatTensor([lens[i]]).to(wav.device))
            if 'beam_search_negative_sampling' in args.method:
                negative_outputs = outputs.clone()

            print(f"beam search time.time() - current 2 : {time.time() - current}")

            if 'beam_search_max' in args.method:
                char_history = pseudo_labels[0].to(args.device)
                if args.certain_only:
                    selected_frame = set()
                    top_idx, top_prob = -1, 0
                    for frame_idx, (output, char_idx) in enumerate(zip(outputs.squeeze(0), char_history)):
                        probs = torch.softmax(output, dim=-1)
                        # print(f"probs[char_idx]: {probs[char_idx]}")
                        if probs[char_idx] > args.prob_threshold:
                            selected_frame.add(frame_idx)
                        if char_idx != blank_index and probs[char_idx].item() > top_prob:
                            top_idx = frame_idx
                            top_prob = probs[char_idx].item()
                    selected_frame.add(top_idx)
                    selected_frame = sorted(selected_frame)
                    print(f"len(selected_frame): {len(selected_frame)}")
                    selected_outputs, selected_char_history = outputs.squeeze(0)[selected_frame], char_history[selected_frame]
                    loss = criterion(selected_outputs / args.temp, selected_char_history)
                else:
                    loss = criterion(outputs / args.temp, char_history)
                (loss / len(wavs)).backward(retain_graph=True)
            elif 'beam_search_all' in args.method:
                loss = 0
                for char_history in pseudo_labels[:args.num_positives]:
                    char_history = char_history.to(args.device)
                    if args.certain_only:
                        selected_frame = set()
                        top_idx, top_prob = -1, 0
                        for frame_idx, (output, char_idx) in enumerate(zip(outputs.squeeze(0), char_history)):
                            probs = torch.softmax(output, dim=-1)
                            if probs[char_idx] > args.prob_threshold:
                                selected_frame.add(frame_idx)
                            if char_idx != blank_index and probs[char_idx].item() > top_prob:
                                top_idx = frame_idx
                                top_prob = probs[char_idx].item()
                        selected_frame.add(top_idx)
                        selected_frame = sorted(selected_frame)
                        selected_outputs, selected_char_history = outputs.squeeze(0)[selected_frame], char_history[selected_frame]
                        loss += criterion(selected_outputs / args.temp, selected_char_history) / len(pseudo_labels)
                    else:
                        loss += criterion(outputs / args.temp, char_history) / len(pseudo_labels)
                (loss / len(wavs)).backward(retain_graph=True)
            if 'beam_search_negative_sampling' in args.method:
                negative_loss = 0
                char_history = pseudo_labels[0].to(args.device)
                if args.negative_sampling_method == "random":
                    for _ in range(args.num_negatives):
                        negative_char_history = torch.randint_like(input=char_history, high=negative_outputs.shape[-1]).to(args.device)
                        negative_mask = (negative_char_history != char_history) & (char_history != 0)

                        selected_frame = []
                        for frame_idx, mask in enumerate(negative_mask):
                            if mask:
                                selected_frame.append(frame_idx)
                        selected_negative_outputs = negative_outputs.squeeze(0)[selected_frame]
                        selected_negative_char_history = negative_char_history[selected_frame]
                        if len(selected_negative_outputs) > 0:
                            negative_loss += -criterion(selected_negative_outputs / args.temp, selected_negative_char_history) / args.num_negatives
                elif args.negative_sampling_method == "beam_candidate":
                    for out_idx in range(len(pseudo_labels))[-args.num_negatives:]:
                        negative_char_history = pseudo_labels[out_idx].to(args.device)
                        negative_mask = (negative_char_history != char_history) & (char_history != 0)

                        selected_frame = []
                        for frame_idx, mask in enumerate(negative_mask):
                            if mask:
                                selected_frame.append(frame_idx)
                        selected_negative_outputs = negative_outputs.squeeze(0)[selected_frame]
                        selected_negative_char_history = negative_char_history[selected_frame]
                        if len(selected_negative_outputs) > 0:
                            negative_loss += -criterion(selected_negative_outputs / args.temp, selected_negative_char_history) / args.num_negatives
                elif args.negative_sampling_method == 'ns3l':
                    negative_mask = torch.where(torch.softmax(negative_outputs, dim=-1) < args.ns_threshold * (10 / negative_outputs.shape[-1]), 1, 0)
                    negative_loss += torch.mean(-torch.log(1 - torch.sum(negative_mask * torch.softmax(negative_outputs / args.temp, dim=-1), dim=-1)))
                if torch.is_tensor(negative_loss):
                    (args.ns_coef * negative_loss / len(wavs)).backward(retain_graph=True)
    if 'diversity_maximization' in args.method:
        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)
            outputs, _ = get_logits_and_pseudo_labels(args, model, decoder_processor, wav, torch.FloatTensor([lens[i]]).to(wav.device))
            predicted_ids = torch.argmax(outputs, dim=-1)
            non_blank = torch.where(predicted_ids != blank_index, 1, 0).bool()
            probs = torch.softmax(outputs[non_blank] / args.temp, dim=-1)
            mean_prob = torch.mean(probs, dim=0)
            loss = torch.sum(mean_prob * torch.log(mean_prob))
            (args.dm_coef * loss / len(wavs)).backward(retain_graph=True)
    if 'renyi_em' in args.method:
        for i, wav in enumerate(wavs):
            wav = wav[:lens[i]].unsqueeze(0)
            outputs, _ = get_logits_and_pseudo_labels(args, model, decoder_processor, wav, torch.FloatTensor([lens[i]]).to(wav.device))
            predicted_ids = torch.argmax(outputs, dim=-1)
            non_blank = torch.where(predicted_ids != blank_index, 1, 0).bool()

            if args.not_blank:
                e_loss = renyi_entropy((outputs / args.temp)[non_blank], alpha=args.renyi_entropy_alpha)
            else:
                e_loss = renyi_entropy(outputs / args.temp, alpha=args.renyi_entropy_alpha)
            (e_loss / (len(wavs))).backward(retain_graph=True)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(args):
    if args.seed:
        set_seed(args.seed)

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

    model = get_model(args)
    original_model = get_model(args)
    if isinstance(model, Wav2Vec2ForCTC): # ctc
        params, _ = collect_params_ctc(model, train_params, bias_only)
    elif isinstance(model, EncoderDecoderASR):
        params, _ = collect_params_attn(model, train_params, bias_only)
    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
        params, _ = collect_params_trans(model, train_params, bias_only)
    optimizer, scheduler = get_optimizer(args, params, opt_name=args.optimizer, lr=lr, scheduler=args.scheduler)
    # if args.use_amp:
    #     torch.set_default_dtype(torch.float16)
    #     model, optimizer = amp.initialize(model, optimizer, opt_level='O3')
    #     global scaler
    #     scaler = GradScaler()
    processor = Wav2Vec2Processor.from_pretrained(args.asr, sampling_rate=sample_rate, return_attention_mask=True) if isinstance(model, Wav2Vec2ForCTC) else None

    global decoder_processor
    if isinstance(model, Wav2Vec2ForCTC): # ctc
        decoder_processor = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")
    elif isinstance(model, EncoderDecoderASR):
        decoder_processor = None
    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
        decoder_processor = BeamRNNTInfer(
            model.decoding.decoding.decoder.to(args.device),
            model.decoding.decoding.joint.to(args.device),
            beam_size=args.beam_width,
            return_best_hypothesis=False,
        )

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
            if args.selective_adaptation:
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
                else:
                    wavs_to_adapt = wavs
                    lens_to_adapt = lens
            else:
                wavs_to_adapt = wavs
                lens_to_adapt = lens

        # if args.use_amp:
        #     wavs_to_adapt = wavs_to_adapt.half()

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
                model = set_rnn_to_train(model)
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