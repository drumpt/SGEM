import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.decoders.seq2seq import S2SRNNGreedySearcher, batch_filter_seq2seq_output
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.common.parts.rnn import label_collate
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses
from nemo.collections.asr.parts.submodules import rnnt_greedy_decoding as greedy_decode
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyBatchedRNNTInfer, pack_hypotheses
# from nemo.collections.common.parts.training_utils import pack_hypotheses

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

# model types are in rnnt_bpe_models.py
original_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from("pretrained_models/stt_en_conformer_transducer_small.nemo").cuda()
original_model.eval()

model = nemo_asr.models.EncDecRNNTBPEModel.restore_from("pretrained_models/stt_en_conformer_transducer_small.nemo").cuda()
with open("trans_arch.txt", "w") as f:
    f.write(str(model))

params, names = [], []
for nm, m in model.named_modules():
    for np, p in m.named_parameters():
        params.append(p)
        names.append(f"{nm}.{np}")
optim, _ = setup_optimizer(params, lr=1e-6)
model_state, optim_state, _ = copy_model_and_optimizer(model, optim, None)

def forward_and_adapt_cr_transducer(input_values, model, optimizer, lens=None):
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

    ce_loss = nn.CrossEntropyLoss()

    encoder_output, encoded_lengths = model(input_signal=input_values.cuda(), input_signal_length=lens.cuda())
    # rnnt_decoder_predictions_tensor -> need to replace
    # best_hyp_texts, _ = model.decoding.rnnt_decoder_predictions_tensor(
    #     encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
    # )


    # Compute hypotheses
    # with torch.inference_mode():
        # hypotheses_list = model.decoding.decoding(
        #     encoder_output=encoder_output, encoded_lengths=encoded_lengths, partial_hypotheses=None
        # )

        # get hypotheses_list
    decoder_training_state = model.decoding.decoding.decoder.training
    joint_training_state = model.decoding.decoding.joint.training

    # Apply optional preprocessing
    encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)
    logitlen = encoded_lengths

    model.decoding.decoding.decoder.eval()
    model.decoding.decoding.joint.eval()

    with model.decoding.decoding.decoder.as_frozen(), model.decoding.decoding.joint.as_frozen():
        inseq = encoder_output  # [B, T, D]
        x, out_len, device = inseq, logitlen, inseq.device
        batchsize = x.shape[0]
        hypotheses = [
            rnnt_utils.Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batchsize)
        ]
        hidden = None

        if model.decoding.decoding.preserve_alignments:
            # alignments is a 3-dimensional dangling list representing B x T x U
            for hyp in hypotheses:
                hyp.alignments = [[]]

        last_label = torch.full([batchsize, 1], fill_value=model.decoding.decoding._blank_index, dtype=torch.long, device=device)
        blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)

        # TTA
        loss = 0

        # Get max sequence length
        max_out_len = out_len.max()
        for time_idx in range(max_out_len):
            f = x.narrow(dim=1, start=time_idx, length=1)  # [B, 1, D]

            not_blank = True
            symbols_added = 0

            blank_mask.mul_(False)
            blank_mask = time_idx >= out_len

            # Start inner loop
            while not_blank and (model.decoding.decoding.max_symbols is None or symbols_added < model.decoding.decoding.max_symbols):
                if time_idx == 0 and symbols_added == 0 and hidden is None:
                    g, hidden_prime = model.decoding.decoding._pred_step(model.decoding.decoding._SOS, hidden, batch_size=batchsize)
                else:
                    g, hidden_prime = model.decoding.decoding._pred_step(last_label, hidden, batch_size=batchsize)

                logp = model.decoding.decoding._joint_step(f, g, log_normalize=None)[:, 0, 0, :]

                # TTA: pseudo labeling
                # probs = F.softmax(logp, dim=-1)
                # max_prob, max_idx = torch.max(probs, dim=-1, keepdim=True)
                # one_hot = torch.FloatTensor(probs.shape).zero_().to('cuda').scatter(1, max_idx, 1)
                # th = 0.95 # threshold
                # if max_prob.item() > th:
                #     loss += ce_loss(logp, one_hot.detach())

                # TTA: entropy minimization
                temperature = 3
                loss += softmax_entropy(logp / temperature, dim=-1).mean()

                if logp.dtype != torch.float32:
                    logp = logp.float()

                # Get index k, of max prob for batch
                v, k = logp.max(1)
                del g

                # Update blank mask with current predicted blanks
                # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                k_is_blank = k == model.decoding.decoding._blank_index

                blank_mask.bitwise_or_(k_is_blank)

                del k_is_blank

                # If preserving alignments, check if sequence length of sample has been reached
                # before adding alignment
                if model.decoding.decoding.preserve_alignments:
                    # Insert logprobs into last timestep per sample
                    logp_vals = logp.to('cpu')
                    logp_ids = logp_vals.max(1)[1]
                    for batch_idx in range(batchsize):
                        if time_idx < out_len[batch_idx]:
                            hypotheses[batch_idx].alignments[-1].append(
                                (logp_vals[batch_idx], logp_ids[batch_idx])
                            )
                    del logp_vals
                del logp

                if blank_mask.all():
                    not_blank = False
                    if model.decoding.decoding.preserve_alignments:
                        # convert Ti-th logits into a torch array
                        for batch_idx in range(batchsize):
                            if len(hypotheses[batch_idx].alignments[-1]) > 0:
                                hypotheses[batch_idx].alignments.append([])  # blank buffer for next timestep
                else:
                    # Collect batch indices where blanks occurred now/past
                    blank_indices = (blank_mask == 1).nonzero(as_tuple=False)

                    # Recover prior state for all samples which predicted blank now/past
                    if hidden is not None:
                        # LSTM has 2 states
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

        optimizer.zero_grad()
        if not isinstance(loss, int):
            loss.requires_grad_(True)
            loss.backward()
        optimizer.step()

        # Remove trailing empty list of alignments at T_{am-len} x Uj
        if model.decoding.decoding.preserve_alignments:
            for batch_idx in range(batchsize):
                if len(hypotheses[batch_idx].alignments[-1]) == 0:
                    del hypotheses[batch_idx].alignments[-1]

        # Preserve states
        for batch_idx in range(batchsize):
            hypotheses[batch_idx].dec_state = model.decoding.decoding.decoder.batch_select_state(hidden, batch_idx)

    # Pack the hypotheses results
    prediction_list = pack_hypotheses(hypotheses, logitlen)

    model.decoding.decoding.decoder.train(decoder_training_state)
    model.decoding.decoding.joint.train(joint_training_state)

    ## upper: in rnnt_greedy_decoding -----
    # hypotheses_list = hypotheses_list[0]
    if isinstance(prediction_list[0], NBestHypotheses):
        hypotheses = []
        all_hypotheses = []

        for nbest_hyp in prediction_list:  # type: NBestHypotheses
            n_hyps = nbest_hyp.n_best_hypotheses  # Extract all hypotheses for this sample
            decoded_hyps = model.decoding.decode_hypothesis(n_hyps)

            # If computing timestamps
            if model.decoding.compute_timestamps is True:
                timestamp_type = model.decoding.cfg.get('rnnt_timestamp_type', 'all')
                for hyp_idx in range(len(decoded_hyps)):
                    decoded_hyps[hyp_idx] = model.decoding.compute_rnnt_timestamps(decoded_hyps[hyp_idx], timestamp_type)

            hypotheses.append(decoded_hyps[0])  # best hypothesis
            all_hypotheses.append(decoded_hyps)

        best_hyp_text = [h.text for h in hypotheses]
        all_hyp_text = [h.text for hh in all_hypotheses for h in hh]
        return best_hyp_text, all_hyp_text

    else:
        hypotheses = model.decoding.decode_hypothesis(prediction_list)

        # If computing timestamps
        if model.decoding.compute_timestamps is True:
            timestamp_type = model.decoding.cfg.get('rnnt_timestamp_type', 'all')
            for hyp_idx in range(len(hypotheses)):
                hypotheses[hyp_idx] = model.decoding.compute_rnnt_timestamps(hypotheses[hyp_idx], timestamp_type)

        best_hyp_text = [h.text for h in hypotheses]
        return best_hyp_text, None

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
    encoded, encoded_len = original_model(input_signal=wavs.cuda(), input_signal_length=lens.cuda())
    best_hyp_texts, _ = original_model.decoding.rnnt_decoder_predictions_tensor(
        encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
    )

    ori_transcription = [best_hyp_text.upper() for best_hyp_text in best_hyp_texts]
    ori_transcriptions += ori_transcription
    ori_wer = wer(list(texts), list(ori_transcription))
    gt_texts += texts
    print(f"\n{i}/{len(dataset)}\noriginal WER: {ori_wer}")

    model, optim, _ = load_model_and_optimizer(model, optim, None, model_state, optim_state, None)
    for i in range(steps):
        model.train()
        forward_and_adapt_cr_transducer(wavs, model, optim, lens=lens)

        if i + 1 in [1, 3, 5, 10, 20, 40]:
            model.eval()
            encoded, encoded_len = model(input_signal=wavs.cuda(), input_signal_length=lens.cuda())
            best_hyp_texts, _ = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
            )
            transcription = [best_hyp_text.upper() for best_hyp_text in best_hyp_texts]
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