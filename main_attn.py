import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.decoders.seq2seq import S2SRNNGreedySearcher, batch_filter_seq2seq_output
import nemo.collections.asr as nemo_asr
from audio_augmentations import *
from jiwer import wer

from data import load_dataset
from main import collect_params, setup_optimizer, softmax_entropy, load_model_and_optimizer, copy_model_and_optimizer

split = ["test-other"]
# dataset_name = "chime"
# dataset_dir = "/home/server08/hdd0/changhun_workspace/CHiME3"
dataset_name = "ted"
dataset_dir = "/home/server08/hdd0/changhun_workspace/TEDLIUM_release2/test"
batch_size=1
extra_noise=0
steps = 10
lr = 2e-6

dataset = load_dataset(split, dataset_name, dataset_dir, batch_size, extra_noise)

original_model = EncoderDecoderASR.from_hparams("speechbrain/asr-crdnn-rnnlm-librispeech", run_opts={"device": "cuda"})
original_model.eval()

model = EncoderDecoderASR.from_hparams("speechbrain/asr-crdnn-rnnlm-librispeech", run_opts={"device": "cuda"})
params, names = [], []
for nm, m in model.named_modules():
    for np, p in m.named_parameters():
        params.append(p)
        names.append(f"{nm}.{np}")
optim, _ = setup_optimizer(params, lr=1e-6) # all
model_state, optim_state, _ = copy_model_and_optimizer(model, optim, None)

def forward_and_adapt_cr_attention(input_values, model, optimizer):
    # weak_transforms = [
    #     RandomApply([PolarityInversion()], p=0.5),
    #     Noise(min_snr=0.001, max_snr=0.005),
    #     # RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.2),
    #     # RandomApply([PitchShift(n_samples=16000*5, sample_rate=16000)], p=0.2),
    #     # RandomApply([Reverb(sample_rate=16000)], p=0.2),
    # ]
    # weak_augmentation = ComposeMany(transforms=weak_transforms, num_augmented_samples=1)

    # strong_transforms = [
    #     RandomApply([PolarityInversion()], p=0.5),
    #     # RandomApply([Noise(min_snr=0.01, max_snr=0.05)], p=0.7),
    #     Noise(min_snr=0.01, max_snr=0.05),
    #     RandomApply([Gain()], p=0.7),
    #     # RandomApply([HighLowPass(sample_rate=16000)], p=0.7),
    #     # RandomApply([PitchShift(n_samples=16000*5, sample_rate=16000)], p=0.7),
    #     RandomApply([Reverb(sample_rate=16000)], p=0.7),
    # ]
    # strong_augmentation = ComposeMany(transforms=strong_transforms, num_augmented_samples=1)

    # ----------

    # adversarial attack

    # transcript = ["Hello"]
    # transcript_len = torch.tensor([5.])

    # from speechbrain.nnet.containers import LengthsCapableSequential
    # encoded = model.mods.encoder(input_values.cuda(), wav_len=torch.tensor([1.]).cuda())
    # decoder, target_length, _ = model.mods.decoder(transcript)
    # joint = model.mods.joint(encoder_outputs=encoded, decoder_outputs=decoder)
    # loss_value = model.mods.loss(
    #     log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
    # )
    # print(loss_value)

    # ----------

    ce_loss = nn.CrossEntropyLoss()
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
    ).to('cuda').train()

    wav_len = torch.tensor([1.0])
    enc_states = model.encode_batch(input_values, wav_lens=wav_len)
    enc_lens = torch.round(enc_states.shape[1] * wav_len).int().cuda()

    device = enc_states.device
    batch_size = enc_states.shape[0]
    memory = greedy_searcher.reset_mem(batch_size, device=device)

    # Using bos as the first input
    inp_tokens = (enc_states.new_zeros(batch_size).fill_(greedy_searcher.bos_index).long())
    log_probs_lst = []
    max_decode_steps = int(enc_states.shape[1] * greedy_searcher.max_decode_ratio)

    loss = 0
    for _ in range(max_decode_steps):
        log_probs, memory, _ = greedy_searcher.forward_step(
            inp_tokens, memory, enc_states, enc_lens
        )
        log_probs_lst.append(log_probs)
        inp_tokens = log_probs.argmax(dim=-1)

        # pseudo labeling
        probs = F.softmax(log_probs, dim=-1)
        max_prob, max_idx = torch.max(probs, dim=-1, keepdim=True)
        one_hot = torch.FloatTensor(probs.shape).zero_().to('cuda').scatter(1, max_idx, 1)
        th = 0.9 # threshold
        if max_prob.item() > th:
            loss += ce_loss(log_probs, one_hot.detach())

        # # entropy minimization
        # temperature = 3
        # loss += softmax_entropy(log_probs / temperature, dim=-1).mean()

    optimizer.zero_grad()
    if not isinstance(loss, int):
        loss.requires_grad_(True)
        loss.backward()
    optimizer.step()

    log_probs = torch.stack(log_probs_lst, dim=1)
    scores, predictions = log_probs.max(dim=-1)
    scores = scores.sum(dim=1).tolist()
    predictions = batch_filter_seq2seq_output(
        predictions, eos_id=greedy_searcher.eos_index
    )
    predicted_words = [
        model.tokenizer.decode_ids(token_seq)
        for token_seq in predictions
    ]
    return predicted_words

# --------------------------------------------------------------------------------------------------------------------------------------------

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

for i, batch in enumerate(dataset):
    lens, wavs, texts, files = batch
    wavs = torch.tensor(wavs)

    ori_transcription, _ = original_model.transcribe_batch(wavs, wav_lens=torch.tensor([1.0]))
    ori_transcriptions += ori_transcription
    ori_wer = wer(list(texts), list(ori_transcription))
    gt_texts += texts
    print(f"\n{i}/{len(dataset)}\noriginal WER: {ori_wer}")

    model, optim, _ = load_model_and_optimizer(model, optim, None, model_state, optim_state, None)
    for i in range(steps):
        model.train()
        forward_and_adapt_cr_attention(wavs, model, optim)

        if i + 1 in [1, 3, 5, 10, 20, 40]:
            model.eval()
            transcription, _ = model.transcribe_batch(wavs, wav_lens=torch.tensor([1.0]))
            ada_wer = wer(list(texts), list(transcription))            
            print(f"adapt-{i + 1} WER: ", ada_wer)
            transcription_list = eval(f"transcriptions_{i + 1}")
            transcription_list += transcription

    del wavs
    torch.cuda.empty_cache()

print("original WER:", wer(gt_texts, ori_transcriptions))
if steps >= 10: 
    print("TTA-1 WER:", wer(gt_texts, transcriptions_1))
    print("TTA-3 WER:", wer(gt_texts, transcriptions_3))
    print("TTA-5 WER:", wer(gt_texts, transcriptions_5))
    print("TTA-10 WER:", wer(gt_texts, transcriptions_10))