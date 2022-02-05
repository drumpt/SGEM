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


def setup_optimizer(params, lr=0.001, beta=0.9, weight_decay=0.):
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
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
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
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    # p.requires_grad = True
        # else: 
        #     for np, p in m.named_parameters():
        #         p.requires_grad = False
            
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
        
    # load dummy dataset and read soundfiles
    librispeech_eval = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    # print(ds)
    # librispeech_eval = load_dataset("librispeech_asr", "other", split="test")
    # next(iter(librispeech_eval))

    def map_to_array(batch):
        speech, _ = sf.read(batch["file"])
        batch["speech"] = speech
        return batch

    librispeech_eval = librispeech_eval.map(map_to_array)

    # tokenize
    # input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest").input_values.cuda()  # Batch size 1

    # set up for tent
    model = configure_model(model)
    params, param_names = collect_params(model)
    optimizer = setup_optimizer(params)

    def map_to_pred(batch):
        input_values = processor(batch["speech"], return_tensors="pt", padding="longest").input_values.cuda()
        # with torch.no_grad():
        #     logits = model(input_values.to("cuda")).logits

           # iteration 
        for _ in range(steps): 
            outputs = forward_and_adapt(input_values, model, optimizer)
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        batch["transcription"] = transcription

        del outputs
        return batch

    result = librispeech_eval.map(map_to_pred, batched=True, batch_size=2)

    # print("WER:", wer(result["text"], result["transcription"]))


    # print(param_names)

 


    # predicted_ids = torch.argmax(outputs, dim=-1)
    # transcription = processor.batch_decode(predicted_ids)







