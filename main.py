from importlib.util import module_for_loader
import os 
import glob
import re
from tqdm import tqdm 
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch import nn
import torch.optim as optim
from jiwer import wer


def setup_optimizer(params, opt_name='Adam', lr=1e-4, beta=0.9, weight_decay=0.):
    opt = getattr(torch.optim, opt_name)
    print(f'[INFO]    optimizer: {opt}')
    if opt_name == 'Adam':
        return opt(params,
                lr=lr,
                betas=(beta, 0.999),
                weight_decay=weight_decay)
    else: 
        return opt(params, lr=lr, weight_decay=weight_decay)


def softmax_entropy(x, dim=2):
    # Entropy of softmax distribution from logits
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

def mcc_loss(x, reweight=True, dim=2, class_num=32):
    
    p = x.softmax(dim) # (1, L, D)
    p = p.squeeze(0) # (L, D)
    if reweight: # (1, L, D) * (L, 1) 
        target_entropy_weight = softmax_entropy(x, dim=2).detach().squeeze(0) # instance-wise entropy (1, L, D)
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight) # (1, L)
        target_entropy_weight = x.shape[1] * target_entropy_weight / torch.sum(target_entropy_weight)
        cov_matrix_t = p.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(p)

    else: 
        cov_matrix_t = p.transpose(1, 0).mm(p) # (D, L) * (L, D) = (D, D)
    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
    return mcc_loss

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    # if nm.split('.')[1] == 'feature_projection':
                    p.requires_grad = True
                    params.append(p)
                    names.append(f"{nm}.{np}")
                
    return params, names


from copy import deepcopy
def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

    return model, optimizer

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

def forward_and_adapt(x, model, optimizer, mask, em_coef=0.9, reweight=False, temp=1., repeat_inference=False):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x).logits / temp
    # adapt
    loss = 0

    if em_coef > 0: 
        try: 
            loss += softmax_entropy(outputs)[mask].mean(0).mean() * em_coef
        except: 
            loss += softmax_entropy(outputs)[mask[:, :-1]].mean(0).mean() * em_coef
    if 1 - em_coef > 0: 
        loss += mcc_loss(outputs, reweight) * (1 - em_coef)
    # print(loss) 
    loss.backward()
    # grad = cal_grad(model)
    # print(grad)
    optimizer.step()
    # optimizer.zero_grad()
    model.zero_grad()

    # inference again
    if repeat_inference:
        with torch.no_grad():
            outputs = model(x).logits
    return outputs


if __name__ == '__main__':
    SAMPLE_RATE = 16000
    asr = "facebook/wav2vec2-base-960h"
    # asr = "facebook/wav2vec2-large-960h-lv60-self"
    steps = 40
    episodic = True
    opt = 'Adam'
    # dataset_dir = '/home/daniel094144/data/LibriSpeech'
    dataset_dir = '/home/daniel094144/data/CHiME3'
    # dataset_name = 'librispeech'
    dataset_name = 'chime'
    # split = 'test-other'

    split = ['et05_bus_real', 'et05_bus_simu', 'et05_caf_real', 'et05_caf_simu', 'et05_ped_simu', 'et05_str_real', 'et05_str_simu']
    lr = 1e-4
    em_coef = 1.
    reweight = False
    batch_size = 1
    temp =  3.
    log_dir = './logs'
    exp_name = dataset_name+'_'+str(em_coef)+'_'+str(steps)+'_'+str(temp)+'_'+asr

    from data import load_dataset
    dataset = load_dataset(split, dataset_name, dataset_dir, batch_size)
    transcriptions = []
    gt_texts = []
    ori_transcriptions = []
    
    # load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained(asr, sampling_rate=SAMPLE_RATE, return_attention_mask=True)
    model = Wav2Vec2ForCTC.from_pretrained(asr).eval().cuda()        
    
    # set up for tent
    model = configure_model(model)
    params, param_names = collect_params(model)
    optimizer = setup_optimizer(params, opt, lr)

    if episodic: 
        model_state, optimizer_state = copy_model_and_optimizer(model, optimizer)

    
    # print(param_names)
    
    for batch in dataset:
        lens, wavs, texts, files = batch
        
        inputs = processor(wavs, return_tensors="pt", padding="longest")
        mask = inputs.attention_mask
        mask = mask[:, ::320][:, :-1]
        
        input_values = inputs.input_values.cuda()
        
        if episodic: 
            model, optimizer = load_model_and_optimizer(model, optimizer, model_state, optimizer_state)
        
        # iteration 
        for i in range(steps): 
            outputs = forward_and_adapt(input_values, model, optimizer, mask.bool(), em_coef, reweight, temp)
            if episodic: 
                if i == 0: 
                    ori = outputs
                    predicted_ids = torch.argmax(ori, dim=-1)
                    ori_transcription = processor.batch_decode(predicted_ids)
                    ori_transcriptions += ori_transcription
                    ori_wer = wer(list(texts), list(ori_transcription))
                    print("original WER:", ori_wer)
            # print(outputs.softmax(2).max(2))
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        ada_wer = wer(list(texts), list(transcription))
        print("adapted WER: ", ada_wer)
        # if ada_wer < ori_wer:
        #     print(texts)
        #     print(ori_transcription)
        #     print(transcription)

        transcriptions += transcription
        gt_texts += texts

    print('------------------------------------')
    print("asr:", asr)
    print("original WER:", wer(gt_texts, ori_transcriptions))
    print("TTA WER:", wer(gt_texts, transcriptions))
    print(f'eposidic? {episodic}')
    print(f'lr = {lr}')
    print(f'optim = {opt}')
    print(f'step = {steps}')
    print(f'em_coef = {em_coef}')
    print(f'reweight = {reweight}')
    print(f'batch size = {batch_size}')
    print(f'temperature = {temp}')
    print('------------------------------------')

    with open(os.path.join(log_dir, exp_name), 'w') as f: 
        f.write(f"original WER: {wer(gt_texts, ori_transcriptions)}\n")
        f.write(f"TTA WER: {wer(gt_texts, transcriptions)}\n")
        f.write(f'eposidic? {episodic}\n')
        f.write(f'lr = {lr}\n')
        f.write(f'optim = {opt}\n')
        f.write(f'step = {steps}\n')
        f.write(f'em_coef = {em_coef}\n')
        f.write(f'reweight = {reweight}\n')
        f.write(f'batch size = {batch_size}\n')
        f.write(f'temperature = {temp}\n')

    









