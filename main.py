import os 
import time
import argparse
import json
from copy import deepcopy
from datetime import datetime
from queue import Queue

import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer
from audio_augmentations import *
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

from data import load_dataset


def setup_optimizer(params, opt_name='AdamW', lr=1e-4, beta=0.9, weight_decay=0., scheduler=None, step_size=1, gamma=0.7):
    opt = getattr(torch.optim, opt_name)
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


def consist_loss(model, input_values, outputs):
    targets = outputs
    # noisy outputs
    model.wav2vec2.encoder.dropout.train()
    noisy_outputs = model(input_values).logits

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
                        div_coef=0, repeat_inference=True):
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

    loss.backward()
    optimizer.step()
    if scheduler is not None: 
        scheduler.step()
    model.zero_grad()

    # inference again
    if repeat_inference:
        with torch.no_grad():
            outputs = model(x).logits
    return outputs


def forward_and_adapt_em_uncertainty(x, model, optimizer, em_coef=0.9, reweight=False, temp=1., not_blank=True, scheduler=None, 
                        div_coef=0, repeat_inference=True):
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
            frame_weight = F.normalize(torch.reciprocal(softmax_entropy(outputs)[non_blank]), dim=0).detach()
            e_loss = torch.sum(frame_weight * softmax_entropy(outputs / temp)[non_blank], dim=0).mean()
        else:
            frame_weight = F.normalize(torch.reciprocal(softmax_entropy(outputs)), dim=0).detach()
            e_loss = torch.sum(frame_weight * softmax_entropy(outputs / temp), dim=0).mean()
        
        loss += e_loss * em_coef

    if  1 - em_coef > 0:
        c_loss = mcc_loss(outputs / temp, reweight)
        loss += c_loss * (1 - em_coef)

    if div_coef > 0: 
        d_loss = div_loss(outputs, not_blank) 
        loss += d_loss * div_coef

    loss.backward()
    optimizer.step()
    if scheduler is not None: 
        scheduler.step()
    model.zero_grad()

    # inference again
    if repeat_inference:
        with torch.no_grad():
            outputs = model(x).logits
    return outputs


def forward_and_adapt_em_sparse(x, model, optimizer, em_coef=0.9, reweight=False, temp=1., not_blank=True, scheduler=None, 
                        div_coef=0, repeat_inference=True):
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
            uncertainty_upper_bound = 0.05
            # entropy_per_frame = softmax_entropy(outputs, dim=2)[non_blank]
            # if len(entropy_per_frame) == 0:
            #     return outputs
            # uncertainty_upper_bound = torch.quantile(entropy_per_frame, 0.5)
            selected_frame = non_blank & torch.where(softmax_entropy(outputs, dim=2) < uncertainty_upper_bound, 1, 0).bool()
            e_loss = softmax_entropy(outputs / temp)[selected_frame].mean(0).mean()
        else:
            uncertainty_upper_bound = 0.05
            # entropy_per_frame = softmax_entropy(outputs, dim=2)
            # uncertainty_upper_bound = torch.quantile(entropy_per_frame, 0.5)
            selected_frame = torch.where(softmax_entropy(outputs, dim=2) < uncertainty_upper_bound, 1, 0).bool()
            e_loss = softmax_entropy(outputs / temp)[selected_frame].mean(0).mean() 
        
        loss += e_loss * em_coef

    if 1 - em_coef > 0: 
        c_loss = mcc_loss(outputs / temp, reweight)
        loss += c_loss * (1 - em_coef)

    if div_coef > 0: 
        d_loss = div_loss(outputs, not_blank) 
        loss += d_loss * div_coef 

    loss.backward()
    optimizer.step()
    if scheduler is not None: 
        scheduler.step()
    model.zero_grad()

    # inference again
    if repeat_inference:
        with torch.no_grad():
            outputs = model(x).logits
    return outputs


def forward_and_adapt_cr(x, model, optimizer, em_coef=0.9, reweight=False, temp=1., not_blank=True, scheduler=None, 
                        div_coef=0, repeat_inference=True, teacher_model=None):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.

    the index of <pad> in vocab is 0
    """
    # kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    ce_loss = nn.CrossEntropyLoss()
    loss = 0

    # from art.attacks.evasion import ImperceptibleASRPyTorch
    # from art.estimators.speech_recognition import PyTorchDeepSpeech
    # import numpy as np
    # attack = ImperceptibleASRPyTorch(PyTorchDeepSpeech(pretrained_model="librispeech"))
    # attack.generate(x.numpy(), np.array(["Hello"]))

    # audio augmentation using fixmatch
    num_chunks = 8
    th = 0.5
    weak_transforms = [
        RandomApply([PolarityInversion()], p=0.1),
        RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.1),
        RandomApply([PitchShift(n_samples=16000*5, sample_rate=16000)], p=0.1),
        RandomApply([Reverb(sample_rate=16000)], p=0.1),
    ]
    weak_augmentation = ComposeMany(transforms=weak_transforms, num_augmented_samples=1)

    strong_transforms = [
        RandomApply([PolarityInversion()], p=0.7),
        RandomApply([Noise(min_snr=0.01, max_snr=0.05)], p=0.7),
        RandomApply([Gain()], p=0.7),
        RandomApply([HighLowPass(sample_rate=16000)], p=0.7),
        RandomApply([PitchShift(n_samples=16000*5, sample_rate=16000)], p=0.7),
        RandomApply([Reverb(sample_rate=16000)], p=0.7),
    ]
    strong_augmentation = ComposeMany(transforms=strong_transforms, num_augmented_samples=1)

    for sub_x in x.chunk(num_chunks, dim=1):
        weak_x = weak_augmentation(sub_x.detach().cpu())

        if teacher_model:
            with torch.no_grad():
                outputs = teacher_model.wav2vec2(weak_x.to('cuda'))
                hidden_states = outputs[0]
                hidden_states = model.dropout(hidden_states)

            weak_outputs = teacher_model(weak_x.to('cuda')).logits

            if len(memory_queue.queue) < n_neighbors:
                weak_probs = F.softmax(weak_outputs, dim=2)
            else:
                pseudo_prob = []
                for hidden_state in hidden_states.squeeze(0):
                    candidates = []
                    for previous_hidden_state, prob in memory_queue.queue:
                        candidates.append((torch.norm(hidden_state - previous_hidden_state), prob))
                    candidates.sort(key=lambda x: x[0])
                    
                    weak_prob_for_one = torch.mean(torch.stack([x[1] for x in candidates[:n_neighbors]]), dim=0)
                    pseudo_prob.append(weak_prob_for_one)
                weak_probs = torch.stack(pseudo_prob).unsqueeze(0)

            for hidden_state, logit in zip(hidden_states.squeeze(0), weak_outputs.squeeze(0)):
                if memory_queue.full():
                    memory_queue.get()
                memory_queue.put((hidden_state, F.softmax(logit, -1)))
        else:
            with torch.no_grad():
                outputs = model.wav2vec2(weak_x.to('cuda'))
                hidden_states = outputs[0]
                hidden_states = model.dropout(hidden_states)
            weak_outputs = model(weak_x.to('cuda')).logits

        confidence, _ = torch.max(weak_probs, dim=2)
        selected_frame = torch.where(confidence > th, 1, 0).bool()

        strong_x = strong_augmentation(sub_x.detach().cpu())
        strong_outputs = model(strong_x.to('cuda')).logits

        # loss += ce_loss(strong_outputs[selected_frame], weak_probs[selected_frame].detach())
        # for i, weak_prob in enumerate(weak_probs):
        #     for strong_output in strong_outputs:
        #         print(strong_output[selected_frame[i]].shape)
        #         print(weak_prob[selected_frame[i]].detach().shape)
        #         loss += ce_loss(
        #             strong_output[selected_frame[i]],
        #             weak_prob[selected_frame[i]].detach()
        #         )

        for _, weak_prob in enumerate(weak_probs):
            for strong_output in strong_outputs:
                loss += ce_loss(
                    strong_output,
                    weak_prob.detach()
                )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    momentum = 0.9
    if teacher_model:
        for teacher_param, student_param in zip(teacher_model.parameters(), model.parameters()):
            with torch.no_grad():
                teacher_param.copy_(momentum * teacher_param + (1 - momentum) * student_param)

    # inference again
    if repeat_inference:
        with torch.no_grad():
            outputs = model(x).logits
    return outputs



if __name__ == '__main__':
    SAMPLE_RATE = 16000
    parser = argparse.ArgumentParser(description="TTA ASR")
    parser.add_argument('--asr', type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--episodic', action='store_true')
    parser.add_argument('--div_coef', type=float, default=0.)
    parser.add_argument('--opt', type=str, default='AdamW')
    parser.add_argument('--dataset_name', type=str, default='librispeech')
    parser.add_argument('--dataset_dir', type=str, default='/mnt/hdd0/changhun_workspace/LibriSpeech')
    parser.add_argument('--split', default=['test-other'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--em_coef', type=float, default=1.)
    parser.add_argument('--reweight', action='store_true')
    parser.add_argument('--bias_only', action='store_true')
    parser.add_argument('--train_feature', action='store_true')
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--temp', type=float, default=2.5)
    parser.add_argument('--non_blank', action='store_true')
    parser.add_argument('--log_dir', type=str, default='./exps')
    parser.add_argument('--extra_noise', type=float, default=0.)
    parser.add_argument('--scheduler', default=None)
    parser.add_argument('--method', default='original')
    parser.add_argument('--teacher_student', action='store_true')

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
    skip_short_thd = None
    train_LN = True
    method = args.method
    teacher_student = args.teacher_student
    exp_name = f"exp_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"

    dataset = load_dataset(split, dataset_name, dataset_dir, batch_size, extra_noise)
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
    print(vars(args))

    # load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained(asr, sampling_rate=SAMPLE_RATE, return_attention_mask=True)

    if teacher_student:
        model = Wav2Vec2ForCTC.from_pretrained(asr).eval().cuda()
        model = configure_model(model)

        teacher_model = Wav2Vec2ForCTC.from_pretrained(asr).eval().cuda()
        student_model = Wav2Vec2ForCTC.from_pretrained(asr).eval().cuda()
        # teacher_model = configure_model(teacher_model)
        # student_model = configure_model(student_model)
        params, param_names = collect_params(student_model, bias_only, train_feature, train_all, train_LN)
        optimizer, scheduler = setup_optimizer(params, opt, lr, scheduler=scheduler)
        if episodic: 
            model_state, optimizer_state, scheduler_state = copy_model_and_optimizer(student_model, optimizer, scheduler)

    else:
        model = Wav2Vec2ForCTC.from_pretrained(asr).eval().cuda()
        # setup for tent
        model = configure_model(model)
        params, param_names = collect_params(model, bias_only, train_feature, train_all, train_LN)
        optimizer, scheduler = setup_optimizer(params, opt, lr, scheduler=scheduler)
        if episodic: 
            model_state, optimizer_state, scheduler_state = copy_model_and_optimizer(model, optimizer, scheduler)

    # import nemo.collections.asr as nemo_asr
    # asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_en_conformer_ctc_large")

    # import inspect
    # print("Hello")    
    # print(inspect.signature(asr_model.forward))
    # print(asr_model)
    # print(type(asr_model))
    # transcriptions = asr_model.transcribe(["file.wav"])
    # object_methods = [method_name for method_name in dir(asr_model) if callable(getattr(object, method_name))]
    # print(object_methods)

    # from speechbrain.pretrained import EncoderDecoderASR
    # asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")

    # # forward-hook
    # mid_getter = MidGetter(model, return_layers={'dropout': 'dropout'}, keep_output=True)
    # activation_list = []
    # id_list = []

    # import json
    # import torch
    # from espnet.nets.pytorch_backend.e2e_asr import E2E
    # from espnet2.bin.asr_inference import Speech2Text

    # speech2text = Speech2Text.from_pretrained(
    #     "kamo-naoyuki/librispeech_asr_train_asr_conformer5_raw_bpe5000_scheduler_confwarmup_steps25000_batch_bins140000000_optim_conflr0.0015_initnone_accum_grad2_sp_valid.acc.ave",
    #     maxlenratio=0.0,
    #     minlenratio=0.0,
    #     beam_size=20,
    #     ctc_weight=0.3,
    #     lm_weight=0.5,
    #     penalty=0.0,
    #     nbest=1
    # )

    # model_dir = "espnet/egs/an4/asr1/exp/train_nodev_pytorch_train_mtlalpha0.5/results"
    # # load model
    # with open(model_dir + "/model.json", "r") as f:
    #     idim, odim, conf = json.load(f)
    # model = E2E.build(idim, odim, **conf)
    # model.load_state_dict(torch.load(model_dir + "/model.acc.best"))
    # model.cpu().eval()
    # vocab = conf["char_list"]
    # print(vocab)
    # model

    queue_maxsize = 256
    n_neighbors = 6

    if method == 'cr':
        memory_queue = Queue(maxsize=queue_maxsize)

    count = 0
    start = time.time()
    for batch in dataset:
        lens, wavs, texts, files = batch
        
        inputs = processor(wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding="longest")
        input_values = inputs.input_values.cuda()
        duration = input_values.shape[1] / SAMPLE_RATE
        durations.append(duration)

        if episodic:
            model, optimizer, scheduler = load_model_and_optimizer(model, optimizer, model_state, optimizer_state, scheduler_state)
        
        # print(torch.tensor([input_values.shape[1]]).to('cuda').shape)
        # kwargs = {"input_signal": input_values.to('cuda'), "input_signal_length": torch.tensor([input_values.shape[1]]).to('cuda')}

        # encoded, encoded_len = asr_model.forward(input_values.to('cuda'), torch.tensor([input_values.shape[1]]).to('cuda'), processed_signal=None, processed_signal_length=None)

        # encoded, encoded_len = self.forward(
        #     input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
        # )
        # best_hyp, all_hyp = self.decoding.rnnt_decoder_predictions_tensor(
        #     encoded,
        #     encoded_len,
        #     return_hypotheses=return_hypotheses,
        #     partial_hypotheses=partial_hypothesis,
        # )

        # print(lens)
        # print(wavs, lens)
        # print(asr_model.encode_batch(input_values, torch.Tensor([1])))
        # print(asr_model.encode_batch(input_values, torch.Tensor([1])).shape)
        # print(asr_model.transcribe_batch(input_values, torch.Tensor([1])))

        # vanilla forward
        with torch.no_grad():
            outputs = model(input_values).logits
        predicted_ids = torch.argmax(outputs, dim=-1)
        ori_transcription = processor.batch_decode(predicted_ids)
        ori_transcriptions += ori_transcription
        ori_wer = wer(list(texts), list(ori_transcription))
        print("\noriginal WER: ", ori_wer)

        if skip_short_thd is not None: 
            if outputs.shape[1] <= skip_short_thd:
                print(f'do not adapt since length is {outputs.shape[1]}')
                count += 1
                continue
        
        # # forward-hook
        # from matplotlib import pyplot as plt
        # import seaborn as sns
        # from sklearn.manifold import TSNE
        # tsne = TSNE(n_components=2)

        # mid_outputs, model_outputs = mid_getter(input_values)
        # predicted_ids = torch.argmax(model_outputs.logits, dim=-1)
        # non_blank_frames = torch.where(predicted_ids != 0, 1, 0).bool()

        # try:
        #     activation_list.extend(mid_outputs['dropout'][non_blank_frames].squeeze(0).detach().cpu().tolist())
        #     id_list.extend(torch.argmax(model_outputs.logits[non_blank_frames].squeeze(0), dim=-1).detach().cpu().tolist())
        #     compressed_list = tsne.fit_transform(activation_list)
        # except:
        #     continue

        # fig = plt.figure(figsize = (10,10))
        # plt.axis('off')
        # plt.legend(list(json.load(open('vocab.json')).keys()))
        # sns.set_style('darkgrid')
        # sns.scatterplot(compressed_list[:,0], compressed_list[:,1], hue=id_list, legend='full')
        # plt.savefig("tsne.png")

        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.title("entropy distribution")
        # plt.hist(softmax_entropy(outputs).squeeze(0).detach().cpu(), color='green', alpha=0.5, edgecolor='black', bins=200)
        # plt.savefig(f"imgs/{files}.png")

        # SUTA
        for i in range(steps):
            if method == 'em_uncertainty':
                outputs = forward_and_adapt_em_uncertainty(input_values, model, optimizer, em_coef, reweight, temp, non_blank, scheduler, div_coef)
            elif method == 'em_sparse':
                outputs = forward_and_adapt_em_sparse(input_values, model, optimizer, em_coef, reweight, temp, non_blank, scheduler, div_coef)
            elif method == 'cr':
                if teacher_student:
                    outputs = forward_and_adapt_cr(input_values, student_model, optimizer, em_coef, reweight, temp, non_blank, scheduler, div_coef, teacher_model=teacher_model)
                else:
                    outputs = forward_and_adapt_cr(input_values, model, optimizer, em_coef, reweight, temp, non_blank, scheduler, div_coef)
            else: # original
                outputs = forward_and_adapt(input_values, model, optimizer, em_coef, reweight, temp, non_blank, scheduler, div_coef)

            # if episodic:
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
                # print(texts, transcription)
                werr = ori_wer - ada_wer
                werrs.append(werr)
                transcriptions_10 += transcription
                
            if i == 19: 
                predicted_ids = torch.argmax(outputs, dim=-1)
                transcription = processor.batch_decode(predicted_ids)
                ada_wer = wer(list(texts), list(transcription))
                # print("adapt-20 WER: ", ada_wer)
                # print(texts, transcription)
                transcriptions_20 += transcription

            if  i == 39: 
                predicted_ids = torch.argmax(outputs, dim=-1)
                transcription = processor.batch_decode(predicted_ids)
                ada_wer = wer(list(texts), list(transcription))
                # print("adapt-40 WER: ", ada_wer)
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

    with open(os.path.join(log_dir, exp_name+".txt"), 'w') as f:
        f.write("=====Hyperparameters=====\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("====================\n")
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
    
    csv_path = os.path.join(log_dir, exp_name+'.csv')
    df = pd.DataFrame({'duration': durations, 'WERR': werrs})
    df.to_csv(csv_path)
    









