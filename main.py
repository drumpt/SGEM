from importlib.util import module_for_loader
import os 
import glob
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
    print(opt)
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

def forward_and_adapt(x, model, optimizer, mask):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x).logits
    # # B, L, D
    # print(outputs.shape, mask.shape)
    # adapt
    try: 
        loss = softmax_entropy(outputs)[mask].mean(0).mean()
    except: 
        loss = softmax_entropy(outputs)[mask[:, :-1]].mean(0).mean()
    # print(loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # inference again
    # with torch.no_grad():
    #     outputs = model(x).logits
    return outputs


############## Build dataset and dataloader ###############
# wav, sr = torchaudio.load(filename)
# wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav).squeeze().cuda()
# input = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors='pt').input_values.cuda()

if __name__ == '__main__':
    SAMPLE_RATE = 16000
    steps = 20
    episodic = True
    opt = 'Adam'
    dataset_dir = '/home/daniel094144/data/LibriSpeech'
    dataset_name = 'librispeech'
    split = 'test-other'
    lr = 1e-4

    # load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", sampling_rate=SAMPLE_RATE, return_attention_mask=True)
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").eval().cuda()        

    # set up for tent
    params, param_names = collect_params(model)
    optimizer = setup_optimizer(params, opt, lr)

    if episodic: 
        model_state, optimizer_state = copy_model_and_optimizer(model, optimizer)

    print(param_names)
    from data import load_dataset
    dataset = load_dataset(split, dataset_name, dataset_dir, 1)
    transcriptions = []
    gt_texts = []
    for batch in dataset:
        lens, wavs, texts, files = batch
        
        inputs = processor(wavs, return_tensors="pt", padding="longest")
        mask = inputs.attention_mask
        mask = mask[:, ::320][:, :-1]

        input_values = inputs.input_values.cuda()
        if episodic: 
            load_model_and_optimizer(model, optimizer, model_state, optimizer_state)
        # iteration 
        for i in range(steps): 
            outputs = forward_and_adapt(input_values, model, optimizer, mask.bool())
            if episodic: 
                if i == 0: 
                    ori = outputs
                    predicted_ids = torch.argmax(ori, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)
                    print("original WER:", wer(list(texts), list(transcription)))

        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        print("adapted WER: ", wer(list(texts), list(transcription)))
        transcriptions += transcription
        gt_texts += texts

    print('------------------------------------')
    print("WER:", wer(gt_texts, transcriptions))
    print(f'eposidic? {episodic}')
    print(f'lr = {lr}')
    print(f'optim = {opt}')
    print(f'step = {steps}')
    print('------------------------------------')






