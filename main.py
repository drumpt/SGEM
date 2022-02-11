import os 
import glob
from tqdm import tqdm 
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
from torch.nn.functional import softmax
from torch import nn
import torch.optim as optim
from jiwer import wer


def setup_optimizer(params, lr=0.0003, beta=0.9, weight_decay=0.):
    return optim.Adam(params,
            lr=lr,
            betas=(beta, 0.999),
            weight_decay=weight_decay)


def softmax_entropy(x, dim=2):
    # Entropy of softmax distribution from logits
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    # model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    # for m in model.modules():
        # if isinstance(m, nn.LayerNorm):
            # m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            # m.track_running_stats = False
            # m.running_mean = None
            # m.running_var = None
    return model

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
                    # if nm.split('.')[-1] == 'final_layer_norm':
                    p.requires_grad = True
                    params.append(p)
                    names.append(f"{nm}.{np}")
            
    return params, names

def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x).logits
    # adapt
    loss = softmax_entropy(outputs).mean(0).mean()
    print(loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # inference again
    with torch.no_grad():
        outputs = model(x).logits
    return outputs


############## Build dataset and dataloader ###############
# wav, sr = torchaudio.load(filename)
# wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav).squeeze().cuda()
# input = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors='pt').input_values.cuda()

if __name__ == '__main__':
    SAMPLE_RATE = 16000
    steps = 1
    # load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", sampling_rate=SAMPLE_RATE)
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").eval().cuda()        

    # set up for tent
    model = configure_model(model)
    params, param_names = collect_params(model)
    optimizer = setup_optimizer(params)
    # # param_grads = [p.requires_grad for p in model.parameters()]
    # print(param_names)
    from data import load_dataset
    dataset = load_dataset('test-other', 'librispeech', '/home/daniel094144/Daniel/data/LibriSpeech', 1)
    transcriptions = []
    gt_texts = []
    for batch in dataset:
        wavs, texts, files = batch
        input_values = processor(wavs, return_tensors="pt", padding="longest").input_values.cuda()

        # iteration 
        for _ in range(steps): 
            outputs = forward_and_adapt(input_values, model, optimizer)

        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        # print(transcription, texts)  
        transcriptions.append(transcription)
        gt_texts.append(texts)

    # print("WER:", wer(result["text"], result["transcription"]))








