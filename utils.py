import random
import os
import logging
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import Wav2Vec2ForCTC
from speechbrain.pretrained import EncoderDecoderASR
import nemo.collections.asr as nemo_asr


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_logger(args):
    logger = logging.getLogger("main")
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
    elif args.asr in ["pretrained_models/stt_en_conformer_ctc_small.nemo", "pretrained_models/stt_en_conformer_ctc_small_ls.nemo"]: # conformers
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.asr).to(args.device).requires_grad_(True).eval()
    elif args.asr in ["pretrained_models/stt_en_conformer_transducer_small.nemo", "pretrained_models/stt_en_conformer_transducer_large.nemo"]: # transducers
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(args.asr).to(args.device).requires_grad_(True).eval()
    return model


def collect_params(model, train_params):
    if isinstance(model, Wav2Vec2ForCTC):
        return collect_params_ctc(model, train_params)
    elif isinstance(model, EncoderDecoderASR):
        return collect_params_attn(model, train_params)
    elif isinstance(model, nemo_asr.models.EncDecCTCModelBPE):
        return collect_params_conf(model, train_params)
    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
        return collect_params_trans(model, train_params)


def collect_params_ctc(model, train_params):
    params, names = [], []
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
                    if np in ['weight', 'bias']:  
                        p.requires_grad = True
                        if not f"{nm}.{np}" in names:
                            params.append(p)
                            names.append(f"{nm}.{np}")
    return params, names


def collect_params_attn(model, train_params):
    params, names = [], []
    for nm, m in model.named_modules():
        for np, p in m.named_parameters():
            collect = False
            if "all" in train_params:
                collect = True
            if 'enc' in train_params and 'encoder' in str(nm):
                collect = True
            if 'dec' in train_params and 'decoder' in str(nm):
                collect = True
            if 'LN' in train_params and isinstance(m, nn.LayerNorm):
                collect = True

            if collect:
                p.requires_grad = True
                params.append(p)
                names.append(f"{nm}.{np}")
    return params, names


def collect_params_conf(model, train_params):
    params, names = [], []
    for nm, m in model.named_modules():
        for np, p in m.named_parameters():
            collect = False 
            if "all" in train_params:
                collect = True
            if 'LN' in train_params and isinstance(m, nn.LayerNorm):
                collect = True
            if 'BN' in train_params and isinstance(m, nn.BatchNorm1d):
                collect = True
            if 'enc' in train_params and 'encoder' in nm:
                collect = True
            if 'dec' in train_params and 'decoder' in nm:
                collect = True

            if collect:
                p.requires_grad = True
                params.append(p)
                names.append(f"{nm}.{np}")
    return params, names


def collect_params_trans(model, train_params):
    params, names = [], []
    for nm, m in model.named_modules():
        for np, p in m.named_parameters():
            collect = False 
            if "all" in train_params:
                collect = True
            if 'LN' in train_params and isinstance(m, nn.LayerNorm):
                collect = True
            if 'BN' in train_params and isinstance(m, nn.BatchNorm1d):
                collect = True
            if 'enc' in train_params and 'encoder' in nm:
                collect = True
            if 'dec' in train_params and 'decoder' in nm:
                collect = True
            if 'joint' in train_params and 'joint' in nm:
                collect = True

            if collect:
                p.requires_grad = True
                params.append(p)
                names.append(f"{nm}.{np}")
    return params, names


def freeze_norm_stats(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.track_running_stats = False


def set_rnn_to_train(model):
    for m in model.modules():
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
        return optimizer, eval(scheduler)(optimizer, T_max=args.t_max, eta_min=args.lr_min)
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
    elif isinstance(model, nemo_asr.models.EncDecCTCModelBPE):
        blank_index = 128
    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
        blank_index = processor.blank
    return blank_index


def softmax_entropy(x, dim=-1):
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def renyi_entropy(x, alpha, dim=-1):
    # x: (B, L, D)
    if alpha == 1:
        return torch.mean(softmax_entropy(x, dim))
    if alpha == 'inf' or alpha == float('inf'):
        entropy, _ = torch.max(x, dim)
        return -torch.mean(torch.log(entropy))
    entropy = torch.log(torch.pow(x.softmax(dim), alpha).sum(dim)) # entropy: B, L
    entropy = entropy / (1 - alpha)
    return torch.mean(entropy)


def non_saturating_loss(x, dim=-1):
    max_idx = torch.argmax(x, dim=dim, keepdim=True)
    one_hots = torch.zeros_like(x).scatter(dim, max_idx, 1).to(x.device)
    return - torch.mean(one_hots * x) + torch.log(((1 - one_hots) * torch.exp(x)).sum(dim=dim)).mean()


def mcc_loss(x, class_num, reweight=False, dim=-1):
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


def log_softmax(x, axis):
    x_max = np.amax(x, axis=axis, keepdims=True)
    if x_max.ndim > 0:
        x_max[~np.isfinite(x_max)] = 0
    elif not np.isfinite(x_max):
        x_max = 0
    tmp = x - x_max
    exp_tmp = np.exp(tmp)
    with np.errstate(divide="ignore"):
        s = np.sum(exp_tmp, axis=axis, keepdims=True)
        out: np.ndarray = np.log(s)
    out = tmp - out
    return out