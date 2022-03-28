from argparse import ArgumentParser
from ast import parse
from importlib.util import module_for_loader
import os 
import glob
import re
from tqdm import tqdm 
import soundfile as sf
import torch
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch import nn
import torch.optim as optim
from jiwer import wer
import pytorch_warmup as warmup


def setup_optimizer(params, opt_name='AdamW', lr=1e-4, beta=0.9, weight_decay=0., scheduler=None, step_size=1, gamma=0.7):
    opt = getattr(torch.optim, opt_name)
    print(f'[INFO]    optimizer: {opt}')
    print(f'[INFO]    scheduler: {scheduler}')
    if opt_name == 'Adam':       
        optimizer = opt(params,
                lr=lr,
                betas=(beta, 0.999),
                weight_decay=weight_decay)
    else: 
        optimizer = opt(params, lr=lr, weight_decay=weight_decay)
    
    if scheduler is not None: 
        return optimizer, eval(scheduler)(optimizer, step_size=step_size, gamma=gamma)
    else: 
        return optimizer, None


def softmax_entropy(x, dim=2):
    # Entropy of softmax distribution from logits
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

def mcc_loss(x, reweight=False, dim=2, class_num=32):
    p = x.softmax(dim) # (1, L, D)
    p = p.squeeze(0) # (L, D)
    if reweight: # (1, L, D) * (L, 1) 
        target_entropy_weight = softmax_entropy(x, dim=2).detach().squeeze(0) # instance-wise entropy (1, L, D)
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight) # (1, L)
        target_entropy_weight = x.shape[1] * target_entropy_weight / torch.sum(target_entropy_weight)
        cov_matrix_t = p.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(p)
    else:    
        cov_matrix_t = p.transpose(1, 0).mm(p) # (D, L) * (L, D) -> (D, D)

    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
   
    return mcc_loss

def div_loss(x, non_blank=None, L_thd=64):
    # maximize entropy of class prediction for every time-step in a utterance 
    # x (1, L, D)
    loss = 0
    x = x.squeeze(0)
    L = x.shape[0]

    if non_blank is not None: 
        cls_pred = x.mean(0)[1:] # (D, )
    else:
        cls_pred = x.mean(0) # (D, )

    loss = -softmax_entropy(cls_pred, 0)

    return loss

def collect_params(model, bias_only=False, train_feature=False, train_all=False, train_LN=True):
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
        print(nm)
        if train_LN: 
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in trainable:  
                        p.requires_grad = True
                        params.append(p)
                        names.append(f"{nm}.{np}")
        if train_feature:
            if len(str(nm).split('.')) > 1:
                if str(nm).split('.')[1] == 'feature_extractor' or str(nm).split('.')[1] == 'feature_projection':
                    for np, p in m.named_parameters():
                        p.requires_grad = True
                        params.append(p)
                        names.append(f"{nm}.{np}")
                        
        if train_all: 
            for np, p in m.named_parameters():
                p.requires_grad = True
                params.append(p)
                names.append(f"{nm}.{np}")
            

    return params, names


import torch.nn.functional as F
# dropout
def consist_loss(model, input_values, outputs):
    targets = outputs
    # noisy outputs
    model.wav2vec2.encoder.dropout.train()
    noisy_outputs = model(input_values).logits

    # loss = F.mse_loss(noisy_outputs, targets, reduction='mean')

    import json
    f = open('vocab.json')
    vocab = json.load(f)


    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=False)
    predicted_ids = torch.argmax(outputs, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    target = []
    for s in transcription:
        if s == ' ':
            s = '|'
        target.append(vocab[s])

    logp = noisy_outputs.log_softmax(1).transpose(1, 0) # L,N,D
    input_len = logp.shape[0]
    tgt_len = len(target)
    loss = ctc_loss(logp, torch.tensor(target).int(), torch.tensor([input_len]), torch.tensor([tgt_len]))
    model.eval()
    return loss


from copy import deepcopy
def copy_model_and_optimizer(model, optimizer, scheduler):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    if scheduler is not None:
        scheduler_state = deepcopy(scheduler.state_dict())
        return model_state, optimizer_state, scheduler_state
    else:
        return model_state, optimizer_state, None

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state, scheduler_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
        return model, optimizer, scheduler
    else: 
        return model, optimizer, None
    

def cal_grad(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def configure_model(model):
    """Configure model for use with tent."""
    model.requires_grad_(False)
    return model

def forward_and_adapt(x, model, optimizer, em_coef=0.9, reweight=False, temp=1., not_blank=True, scheduler=None, 
                        div_coef=0, repeat_inference=True, skip_short_thd=None):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.

    the index of <pad> in vocab is 0
    """
    # forward
    outputs = model(x).logits
    
    predicted_ids = torch.argmax(outputs, dim=-1)
    non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
    # adapt
    loss = 0

    if em_coef > 0: 
        if not_blank:      
            e_loss = softmax_entropy(outputs / temp)[non_blank].mean(0).mean()
    
        else: 
            e_loss = softmax_entropy(outputs / temp).mean(0).mean() 
        
        loss += e_loss * em_coef

    if 1 - em_coef > 0: 
        c_loss = mcc_loss(outputs / temp, reweight)
        loss += c_loss * (1 - em_coef)

    if div_coef > 0: 
        d_loss = div_loss(outputs, not_blank) 
        loss += d_loss * div_coef 
    # print(f'e_loss = {e_loss}; c_loss = {c_loss}; d_loss = {d_loss}') 
    # con_coef = 0.5
    # print(consist_loss(model, x, outputs))
    # loss = loss * (1-con_coef) + consist_loss(model, x, outputs) * con_coef

    loss.backward()
    # grad = cal_grad(model)
    # print(grad)
    optimizer.step()
    if scheduler is not None: 
        scheduler.step()
    # optimizer.zero_grad()
    model.zero_grad()

    # inference again
    if repeat_inference:
        with torch.no_grad():
            outputs = model(x).logits
    return outputs

import argparse

if __name__ == '__main__':
    SAMPLE_RATE = 16000
    parser = argparse.ArgumentParser(description="TTA ASR")
    parser.add_argument('--asr', type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--episodic', action='store_true')
    parser.add_argument('--div_coef', type=float, default=0.)
    parser.add_argument('--opt', type=str, default='AdamW')
    parser.add_argument('--dataset_name', type=str, default='librispeech')
    parser.add_argument('--dataset_dir', type=str, default='/home/daniel094144/data/LibriSpeech')
    parser.add_argument('--split', default=['test-other'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--em_coef', type=float, default=1.)
    parser.add_argument('--reweight', action='store_true')
    parser.add_argument('--bias_only', action='store_true')
    parser.add_argument('--enhance', action='store_true')
    parser.add_argument('--train_feature', action='store_true')
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--temp', type=float, default=2.5)
    parser.add_argument('--non_blank', action='store_true')
    parser.add_argument('--log_dir', type=str, default='./exps')
    parser.add_argument('--extra_noise', type=float, default=0.)
    parser.add_argument('--scheduler', default=None)
    
    
    # asr = "facebook/wav2vec2-base-960h"
    # asr = "facebook/wav2vec2-large-960h-lv60-self"
    # asr = "facebook/wav2vec2-large-960h-lv60"
    # asr = "facebook/wav2vec2-large-robust-ft-swbd-300h"

    args = parser.parse_args()
    asr = args.asr
    steps = args.steps
    episodic = args.episodic
    opt = args.opt
    dataset_dir = args.dataset_dir
    dataset_name = args.dataset_name
    split = args.split
    lr = args.lr
    em_coef = args.em_coef
    reweight = args.reweight
    batch_size = args.batch_size
    temp =  args.temp
    non_blank = args.non_blank
    log_dir = args.log_dir
    extra_noise = args.extra_noise
    scheduler = args.scheduler
    div_coef = args.div_coef
    bias_only = args.bias_only
    train_feature = args.train_feature
    train_all = args.train_all
    enhance = args.enhance
    skip_short_thd = None
    train_LN = True

    exp_name = dataset_name+'_'+str(em_coef)+'_'+str(steps)+'_'+str(temp)+'_'+asr.split('/')[-1]+'_'+'non_blank'+str(non_blank)+'_noise_'+str(extra_noise)+'_rew_'+str(reweight)+'_div_'+str(div_coef)+'_bias_'+str(bias_only)+'_feat_'+str(train_feature)+'_se_'+str(enhance)+'_all_'+str(train_all)+'_LN_'+str(train_LN)

    from data import load_dataset
    dataset = load_dataset(split, dataset_name, dataset_dir, batch_size, extra_noise, enhance)
    transcriptions_1 = []
    transcriptions_3 = []
    transcriptions_5 = []
    transcriptions_10 = []
    transcriptions_20 = []
    transcriptions_40 = []
    gt_texts = []
    ori_transcriptions = []
    durations = []
    werrs = []

    print('------------------------------------')
    print(f'exp: {exp_name}')
    print(f'eposidic? {episodic}')
    print(f'lr = {lr}')
    print(f'optim = {opt}')
    print(f'step = {steps}')
    print(f'em_coef = {em_coef}')
    print(f'reweight = {reweight}')
    print(f'batch size = {batch_size}')
    print(f'temperature = {temp}')
    print(f'non_blank = {str(non_blank)}')
    print(f'extra_noise = {extra_noise}')
    print(f'scheduler = {str(scheduler)}')
    print(f'div_coef = {str(div_coef)}')
    print(f'bias_only = {bias_only}')
    print(f'train_feature = {train_feature}')
    print(f'train_all = {train_all}')
    print(f'train_LN = {train_LN}')
    print(f'enhance = {str(enhance)}')

    # load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained(asr, sampling_rate=SAMPLE_RATE, return_attention_mask=True)
    model = Wav2Vec2ForCTC.from_pretrained(asr).eval().cuda()        
    
    # set up for tent
    model = configure_model(model)
    params, param_names = collect_params(model, bias_only, train_feature, train_all, train_LN)
    optimizer, scheduler = setup_optimizer(params, opt, lr, scheduler=scheduler)

    if episodic: 
        model_state, optimizer_state, scheduler_state = copy_model_and_optimizer(model, optimizer, scheduler)

    
    print(param_names)
    count = 0

    import time
    start = time.time()
    for batch in dataset:
        lens, wavs, texts, files = batch
        
        inputs = processor(wavs, return_tensors="pt", padding="longest")
        input_values = inputs.input_values.cuda()
        duration = input_values.shape[1] / SAMPLE_RATE
        durations.append(duration)
        
        if episodic: 
            model, optimizer, scheduler = load_model_and_optimizer(model, optimizer, model_state, optimizer_state, scheduler_state)
        
        # vanilla forward 
        with torch.no_grad():
            outputs = model(input_values).logits
        predicted_ids = torch.argmax(outputs, dim=-1)
        ori_transcription = processor.batch_decode(predicted_ids)
        ori_transcriptions += ori_transcription
        ori_wer = wer(list(texts), list(ori_transcription))
        print("original WER: ", ori_wer)

        
        if skip_short_thd is not None: 
            if outputs.shape[1] <= skip_short_thd:
                print(f'do not adapt since length is {outputs.shape[1]}')
                count += 1
                continue
        
        # SUTA
        for i in range(steps): 
            outputs = forward_and_adapt(input_values, model, optimizer, em_coef, reweight, temp, non_blank, scheduler, div_coef)
            if episodic: 
                if i == 0: 
                    predicted_ids = torch.argmax(outputs, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)
                    ada_wer = wer(list(texts), list(transcription))
                    print("adapt-1 WER:  ", ada_wer)
                    # print(texts, transcription)
                    transcriptions_1 += transcription

                if i == 2: 
                    predicted_ids = torch.argmax(outputs, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)
                    ada_wer = wer(list(texts), list(transcription))
                    print("adapt-3 WER:  ", ada_wer)
                    # print(texts, transcription)
                    transcriptions_3 += transcription

                if i == 4: 
                    predicted_ids = torch.argmax(outputs, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)
                    ada_wer = wer(list(texts), list(transcription))
                    print("adapt-5 WER:  ", ada_wer)
                    # print(texts, transcription)
                    transcriptions_5 += transcription

                if i == 9: 
                    predicted_ids = torch.argmax(outputs, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)
                    ada_wer = wer(list(texts), list(transcription))
                    print("adapt-10 WER: ", ada_wer)
                    werr = ori_wer - ada_wer
                    werrs.append(werr)
                    # print(texts, transcription)
                    transcriptions_10 += transcription
                    
                if i == 19: 
                    predicted_ids = torch.argmax(outputs, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)
                    ada_wer = wer(list(texts), list(transcription))
                    print("adapt-20 WER: ", ada_wer)
                    # print(texts, transcription)
                    transcriptions_20 += transcription

                if  i == 39: 
                    predicted_ids = torch.argmax(outputs, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)
                    ada_wer = wer(list(texts), list(transcription))
                    print("adapt-40 WER: ", ada_wer)
                    # print(texts, transcription)
                    transcriptions_40 += transcription
        
        del input_values
        torch.cuda.empty_cache()
        gt_texts += texts


    print("asr:", asr)
    print(f'non-adapted count = {count}')
    print(f'dataset num = {len(dataset)}')
    print("original WER:", wer(gt_texts, ori_transcriptions))
    if steps >= 10: 
        print("TTA-1 WER:", wer(gt_texts, transcriptions_1))
        print("TTA-3 WER:", wer(gt_texts, transcriptions_3))
        print("TTA-5 WER:", wer(gt_texts, transcriptions_5))
        print("TTA-10 WER:", wer(gt_texts, transcriptions_10))
    if steps >= 20: 
        print("TTA-20 WER:", wer(gt_texts, transcriptions_20))
    if steps >= 40:
        print("TTA-40 WER:", wer(gt_texts, transcriptions_40))
    print('------------------------------------')


    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with open(os.path.join(log_dir, exp_name), 'w') as f: 
        f.write(f"original WER: {wer(gt_texts, ori_transcriptions)}\n")
        if steps >= 10: 
            f.write(f"TTA-1 WER: {wer(gt_texts, transcriptions_1)}\n")
            f.write(f"TTA-3 WER: {wer(gt_texts, transcriptions_3)}\n")
            f.write(f"TTA-5 WER: {wer(gt_texts, transcriptions_5)}\n")
            f.write(f"TTA-10 WER: {wer(gt_texts, transcriptions_10)}\n")
        if steps >= 20:
            f.write(f"TTA-20 WER: {wer(gt_texts, transcriptions_20)}\n")
        if steps >= 40:
            f.write(f"TTA-40 WER: {wer(gt_texts, transcriptions_40)}\n")
        f.write(f'eposidic? {episodic}\n')
        f.write(f'lr = {lr}\n')
        f.write(f'optim = {opt}\n')
        f.write(f'step = {steps}\n')
        f.write(f'em_coef = {em_coef}\n')
        f.write(f'reweight = {reweight}\n')
        f.write(f'batch size = {batch_size}\n')
        f.write(f'temperature = {temp}\n')
        f.write(f'non_blank = {str(non_blank)}\n')
        f.write(f'extra_noise = {extra_noise}\n')
        f.write(f'scheduler = {str(scheduler)}\n')
        f.write(f'div_coef = {str(div_coef)}\n')
        f.write(f'bias_only = {str(bias_only)}\n')
        f.write(f'train_feature = {str(train_feature)}\n')
        f.write(f'train_all = {str(train_all)}\n')
        f.write(f'train_LN = {str(train_LN)}\n')
        f.write(f'enhance = {str(enhance)}\n')
    
    csv_path = os.path.join(log_dir, exp_name+'.csv')
    df = pd.DataFrame({'duration': durations, 'WERR': werrs})
    df.to_csv(csv_path)
    









