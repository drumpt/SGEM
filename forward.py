import time
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import heapq
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast

from transformers import Wav2Vec2ForCTC
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.decoders.ctc import CTCPrefixScorer
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.common.parts.rnn import label_collate
from pyctcdecode.alphabet import BPE_TOKEN
from pyctcdecode.constants import (
    DEFAULT_HOTWORD_WEIGHT,
    DEFAULT_MIN_TOKEN_LOGP,
    DEFAULT_PRUNE_BEAMS,
    DEFAULT_PRUNE_LOGP,
    MIN_TOKEN_CLIP_P,
)
from pyctcdecode.language_model import HotwordScorer
try:
    import kenlm
except ImportError:
    pass


Frames = Tuple[int, int]
WordFrames = Tuple[str, Frames]
LMBeam = Tuple[str, str, str, Optional[str], List[Frames], Frames, float, float]
LMState = Optional[Union["kenlm.State", List["kenlm.State"]]]
OutputBeam = Tuple[str, LMState, List[WordFrames], float, float]
OutputBeamMPSafe = Tuple[str, List[WordFrames], float, float]
NULL_FRAMES: Frames = (-1, -1)  # placeholder that gets replaced with positive integer frame indices
EMPTY_START_BEAM = ("", "", "", None, [], NULL_FRAMES, 0.0, [])



@dataclass
class Hypothesis:
    score: float
    y_sequence: Union[List[int], torch.Tensor]
    text: Optional[str] = None
    dec_out: Optional[List[torch.Tensor]] = None
    dec_state: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor]]] = None
    timestep: Union[List[int], torch.Tensor] = field(default_factory=list)
    alignments: Optional[Union[List[int], List[List[int]]]] = None
    length: Union[int, torch.Tensor] = 0
    y: List[torch.tensor] = None
    lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None
    lm_scores: Optional[torch.Tensor] = None
    tokens: Optional[Union[List[int], torch.Tensor]] = None
    last_token: Optional[torch.Tensor] = None
    token_list: List = field(default_factory=list)
    # logit_list: List[torch.Tensor] = field(default_factory=list)


@torch.no_grad()
def transcribe_batch(args, model, processor, wavs, lens):
    transcription = []
    if isinstance(model, Wav2Vec2ForCTC):
        for wav, len in zip(wavs, lens):
            wav = wav[:len].unsqueeze(0)
            outputs = model(wav).logits
            predicted_ids = torch.argmax(outputs, dim=-1)
            text = processor.batch_decode(predicted_ids)
            transcription.append(text[0])
    elif isinstance(model, EncoderDecoderASR):
        for wav, len in zip(wavs, lens):
            wav = wav[:len].unsqueeze(0)
            text = model.transcribe_batch(wav, wav_lens=torch.ones(1).to(args.device))[0]
            transcription.append(text[0])
    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
        for wav, len in zip(wavs, lens):
            wav = wav[:len].unsqueeze(0)
            len = len.unsqueeze(0)

            encoded_feature, encoded_len = model(input_signal=wav, input_signal_length=len)
            best_hyp_texts, _ = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded_feature, encoded_lengths=encoded_len, return_hypotheses=False
            )
            text = [best_hyp_text.upper() for best_hyp_text in best_hyp_texts][0]
            transcription.append(text)
    return transcription


def forward_batch(args, model, wavs, lens, labels=None):
    if isinstance(model, Wav2Vec2ForCTC):
        outputs = forward_ctc(args, model, wavs, lens, labels)
    elif isinstance(model, EncoderDecoderASR):
        outputs = forward_attn(args, model, wavs, lens, labels)
    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
        outputs = forward_trans(args, model, wavs, lens, labels)
    return outputs


def forward_ctc(args, model, wavs, lens, labels):
    # TODO: implement beam-search based logits for entropy minimization
    logits = model(wavs).logits
    return logits


def forward_attn(args, model, wavs, lens, labels):
    def decoder_forward_step(model, inp_tokens, memory, enc_states, enc_lens):
        """Performs a step in the implemented beamsearcher."""
        hs, c = memory
        e = model.emb(inp_tokens)
        dec_out, hs, c, w = model.dec.forward_step(
            e, hs, c, enc_states, enc_lens
        )
        log_probs = model.softmax(model.fc(dec_out) / model.temperature)

        if model.dec.attn_type == "multiheadlocation":
            w = torch.mean(w, dim=1)
        return log_probs, (hs, c), w

    logits = []
    enc_states = model.encode_batch(wavs, lens)
    enc_lens = torch.tensor([enc_states.shape[1]]).to(args.device)

    device = enc_states.device
    batch_size = enc_states.shape[0]
    memory = model.mods.decoder.reset_mem(batch_size, device=device)

    inp_tokens = (enc_states.new_zeros(batch_size).fill_(model.mods.decoder.bos_index).long())
    max_decode_steps = int(enc_states.shape[1] * model.mods.decoder.max_decode_ratio)

    for decode_step in range(max_decode_steps):
        log_probs, memory, _ = decoder_forward_step(model.mods.decoder, inp_tokens, memory, enc_states, enc_lens)
        logits.append(log_probs)
        # teacher-forcing using beam search
        if labels != None:
            inp_tokens = torch.tensor([labels[decode_step]]).to(log_probs.device)
        else:
            inp_tokens = log_probs.argmax(dim=-1)
    logits = torch.stack(logits, dim=1).to(args.device)
    return logits


def forward_trans(args, model, wavs, lens, labels):
    logits = []
    encoder_output, encoded_lengths = model(input_signal=wavs, input_signal_length=lens)
    encoder_output = encoder_output.transpose(1, 2)
    logitlen = encoded_lengths

    inseq = encoder_output  # [B, T, D]
    x, out_len, device = inseq, logitlen, inseq.device
    batchsize = x.shape[0]
    hypotheses = [rnnt_utils.Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batchsize)]
    hidden = None

    if model.decoding.decoding.preserve_alignments:
        for hyp in hypotheses:
            hyp.alignments = [[]]

    last_label = torch.full([batchsize, 1], fill_value=model.decoding.decoding._blank_index, dtype=torch.long, device=device)
    blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)

    max_out_len = out_len.max()
    for time_idx in range(max_out_len):
        f = x.narrow(dim=1, start=time_idx, length=1)  # [B, 1, D]

        not_blank = True
        symbols_added = 0

        blank_mask.mul_(False)
        blank_mask = time_idx >= out_len

        while not_blank and (model.decoding.decoding.max_symbols is None or symbols_added < model.decoding.decoding.max_symbols):
            if time_idx == 0 and symbols_added == 0 and hidden is None:
                in_label = model.decoding.decoding._SOS
            else:
                in_label = last_label
            if isinstance(in_label, torch.Tensor) and in_label.dtype != torch.long:
                in_label = in_label.long()
                g, hidden_prime = model.decoding.decoding.decoder.predict(None, hidden, False, batchsize)
            else:
                if in_label == model.decoding.decoding._SOS:
                    g, hidden_prime = model.decoding.decoding.decoder.predict(None, hidden, False, batchsize)
                else:
                    in_label = label_collate([[in_label.cpu()]])
                    g, hidden_prime = model.decoding.decoding.decoder.predict(in_label, hidden, False, batchsize)

            logp = model.decoding.decoding.joint.joint(f, g)
            if not logp.is_cuda:
                logp = logp.log_softmax(dim=len(logp.shape) - 1)
            logp = logp[:, 0, 0, :]

            if logp.dtype != torch.float32:
                logp = logp.float()

            # teacher-forcing using beam search
            if labels != None:
                label_idx = len(logits)
                label = labels[label_idx] if label_idx < len(labels) else model.decoding.decoding._blank_index
                v, k = logp[:, label], torch.tensor([label for _ in range(logp.shape[0])]).to(logp.device)
            else:
                v, k = logp.max(1)
            del g

            logits.append(logp)

            k_is_blank = k == model.decoding.decoding._blank_index
            blank_mask.bitwise_or_(k_is_blank)
            del k_is_blank

            if model.decoding.decoding.preserve_alignments:
                logp_vals = logp.to('cpu')
                logp_ids = logp_vals.max(1)[1]
                for batch_idx in range(batchsize):
                    if time_idx < out_len[batch_idx]:
                        hypotheses[batch_idx].alignments[-2].append(
                            (logp_vals[batch_idx], logp_ids[batch_idx])
                        )
                del logp_vals

            if blank_mask.all():
                not_blank = False
                if model.decoding.decoding.preserve_alignments:
                    for batch_idx in range(batchsize):
                        if len(hypotheses[batch_idx].alignments[-2]) > 0:
                            hypotheses[batch_idx].alignments.append([])  # blank buffer for next timestep
            else:
                blank_indices = (blank_mask == 1).nonzero(as_tuple=False)
                if hidden is not None:
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
    logits = torch.stack(logits, dim=1)[:, :max_out_len, :]
    return logits


def get_logits_and_pseudo_labels(args, model, processor, wavs, lens):
    if args.decoding_method == "greedy_search" or args.beam_width == 1: # greedy search
        logits = forward_batch(args, model, wavs, lens)
        pseudo_labels = [torch.argmax(logits, dim=-1).squeeze(0)]
    else: # beam search
        encoder_output, encoder_length = encode_batch(args, model, wavs, lens)
        pseudo_labels = decode_batch(args, model, processor, encoder_output, encoder_length)
        logits = forward_batch(args, model, wavs, lens, labels=pseudo_labels[0])
    # print(f"logits.shape: {logits.shape}")
    # print(f"pseudo_labels[0].shape: {pseudo_labels[0].shape}")
    return logits, pseudo_labels


@torch.no_grad()
def encode_batch(args, model, wavs, lens):
    if isinstance(model, Wav2Vec2ForCTC):
        logits = model(wavs).logits
        logitlen = torch.tensor([logits.shape[1]]).to(logits.device)
        outputs = logits, logitlen
    elif isinstance(model, EncoderDecoderASR):
        enc_states = model.encode_batch(wavs, lens)
        enc_lens = torch.tensor([enc_states.shape[1]]).to(args.device)
        outputs = enc_states, enc_lens
    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
        enc_states, enc_lens = model(input_signal=wavs, input_signal_length=lens)
        enc_states = enc_states.transpose(1, 2)
        outputs = enc_states, enc_lens
    return outputs


@torch.no_grad()
def decode_batch(args, model, processor, encoder_output, encoder_length):
    def _log_softmax(x, axis):
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

    beam_width = args.beam_width if args.decoding_method == "beam_search" else 1
    if isinstance(model, Wav2Vec2ForCTC):
        logits = encoder_output.squeeze(0).detach().cpu().numpy()
        logits = np.clip(_log_softmax(logits, axis=1), np.log(MIN_TOKEN_CLIP_P), 0)
        pseudo_labels = decode_ctc(
            processor.decoder,
            logits=logits,
            beam_width=beam_width,
            beam_prune_logp=DEFAULT_PRUNE_LOGP,
            token_min_logp=DEFAULT_MIN_TOKEN_LOGP,
            prune_history=DEFAULT_PRUNE_BEAMS,
            hotword_scorer=HotwordScorer.build_scorer(None, weight=DEFAULT_HOTWORD_WEIGHT),
            lm_start_state=None,
        )
    elif isinstance(model, EncoderDecoderASR):
        model.mods.decoder.topk = beam_width 
        pseudo_labels = decode_attn(
            model.mods.decoder,
            encoder_output,
            torch.FloatTensor([encoder_length]).to(encoder_output.device),
            beam_width,
        )
    elif isinstance(model, nemo_asr.models.EncDecRNNTBPEModel):
        processor.beam_size = beam_width
        pseudo_labels = decode_trans(processor, encoder_output, encoder_length)
    return pseudo_labels


def decode_ctc(
    model,
    logits,
    beam_width,
    beam_prune_logp,
    token_min_logp,
    prune_history,
    hotword_scorer,
    lm_start_state,
):
    def _merge_beams(beams):
        """Merge beams with same prefix together."""
        beam_dict = {}
        for text, next_word, word_part, last_char, text_frames, part_frames, logit_score, idx_history in beams:
            new_text = _merge_tokens(text, next_word)
            hash_idx = (new_text, word_part, last_char)
            if hash_idx not in beam_dict:
                beam_dict[hash_idx] = (
                    text,
                    next_word,
                    word_part,
                    last_char,
                    text_frames,
                    part_frames,
                    logit_score,
                    idx_history
                )
            else:
                beam_dict[hash_idx] = (
                    text,
                    next_word,
                    word_part,
                    last_char,
                    text_frames,
                    part_frames,
                    _sum_log_scores(beam_dict[hash_idx][-2], logit_score),
                    idx_history
                )
        return list(beam_dict.values())


    def _sort_and_trim_beams(beams, beam_width: int):
        """Take top N beams by score."""
        return heapq.nlargest(beam_width, beams, key=lambda x: x[-2])


    def _merge_tokens(token_1: str, token_2: str) -> str:
        """Fast, whitespace safe merging of tokens."""
        if len(token_2) == 0:
            text = token_1
        elif len(token_1) == 0:
            text = token_2
        else:
            text = token_1 + " " + token_2
        return text


    def _sum_log_scores(s1: float, s2: float) -> float:
        """Sum log odds in a numerically stable way."""
        # this is slightly faster than using max
        if s1 >= s2:
            log_sum = s1 + math.log(1 + math.exp(s2 - s1))
        else:
            log_sum = s2 + math.log(1 + math.exp(s1 - s2))
        return log_sum

    def get_new_beams(
        model,
        beams,
        idx_list,
        frame_idx,
        logit_col,
    ):
        new_beams = []
        # bpe we can also have trailing word boundaries ▁⁇▁ so we may need to remember breaks
        force_next_break = False
        for idx_char in idx_list:
            p_char = logit_col[idx_char]
            char = model._idx2vocab[idx_char]
            for (
                text,
                next_word,
                word_part,
                last_char,
                text_frames,
                part_frames,
                logit_score,
                idx_history,
            ) in beams:
                if char == "" or last_char == char:
                    if char == "":
                        new_end_frame = part_frames[0]
                    else:
                        new_end_frame = frame_idx + 1
                    new_part_frames = (
                        part_frames if char == "" else (part_frames[0], new_end_frame)
                    )
                    new_beams.append(
                        (
                            text,
                            next_word,
                            word_part,
                            char,
                            text_frames,
                            new_part_frames,
                            logit_score + p_char,
                            idx_history + [idx_char],
                        )
                    )
                # if bpe and leading space char
                elif model._is_bpe and (char[:1] == BPE_TOKEN or force_next_break):
                    force_next_break = False
                    # some tokens are bounded on both sides like ▁⁇▁
                    clean_char = char
                    if char[:1] == BPE_TOKEN:
                        clean_char = clean_char[1:]
                    if char[-1:] == BPE_TOKEN:
                        clean_char = clean_char[:-1]
                        force_next_break = True
                    new_frame_list = (
                        text_frames if word_part == "" else text_frames + [part_frames]
                    )
                    new_beams.append(
                        (
                            text,
                            word_part,
                            clean_char,
                            char,
                            new_frame_list,
                            (frame_idx, frame_idx + 1),
                            logit_score + p_char,
                            idx_history + [idx_char],
                        )
                    )
                # if not bpe and space char
                elif not model._is_bpe and char == " ":
                    new_frame_list = (
                        text_frames if word_part == "" else text_frames + [part_frames]
                    )
                    new_beams.append(
                        (
                            text,
                            word_part,
                            "",
                            char,
                            new_frame_list,
                            NULL_FRAMES,
                            logit_score + p_char,
                            idx_history + [idx_char],
                        )
                    )
                # general update of continuing token without space
                else:
                    new_part_frames = (
                        (frame_idx, frame_idx + 1)
                        if part_frames[0] < 0
                        else (part_frames[0], frame_idx + 1)
                    )
                    new_beams.append(
                        (
                            text,
                            next_word,
                            word_part + char,
                            char,
                            text_frames,
                            new_part_frames,
                            logit_score + p_char,
                            idx_history + [idx_char],
                        )
                    )
        new_beams = _merge_beams(new_beams)
        return new_beams

    def get_lm_beams(
        model,
        beams,
        hotword_scorer: HotwordScorer,
        cached_lm_scores: Dict[str, Tuple[float, float, LMState]],
        cached_partial_token_scores: Dict[str, float],
        is_eos: bool = False,
    ) -> List[LMBeam]:
        """Update score by averaging logit_score and lm_score."""
        # get language model and see if exists
        language_model = model._language_model

        # if no language model available then return raw score + hotwords as lm score
        if language_model is None:
            new_beams = []
            for text, next_word, word_part, last_char, frame_list, frames, logit_score, idx_history in beams:
                new_text = _merge_tokens(text, next_word)
                # note that usually this gets scaled with alpha
                lm_hw_score = (
                    logit_score
                    + hotword_scorer.score(new_text)
                    + hotword_scorer.score_partial_token(word_part)
                )
                new_beams.append(
                    (
                        new_text,
                        "",
                        word_part,
                        last_char,
                        frame_list,
                        frames,
                        logit_score,
                        lm_hw_score,
                        idx_history
                    )
                )
            return new_beams

        new_beams = []
        for text, next_word, word_part, last_char, frame_list, frames, logit_score, idx_history in beams:
            new_text = _merge_tokens(text, next_word)
            if new_text not in cached_lm_scores:
                _, prev_raw_lm_score, start_state = cached_lm_scores[text]
                score, end_state = language_model.score(start_state, next_word, is_last_word=is_eos)
                raw_lm_score = prev_raw_lm_score + score
                lm_hw_score = raw_lm_score + hotword_scorer.score(new_text)
                cached_lm_scores[new_text] = (lm_hw_score, raw_lm_score, end_state)
            lm_score, _, _ = cached_lm_scores[new_text]

            if len(word_part) > 0:
                if word_part not in cached_partial_token_scores:
                    # if prefix available in hotword trie use that, otherwise default to char trie
                    if word_part in hotword_scorer:
                        cached_partial_token_scores[word_part] = hotword_scorer.score_partial_token(
                            word_part
                        )
                    else:
                        cached_partial_token_scores[word_part] = language_model.score_partial_token(
                            word_part
                        )
                lm_score += cached_partial_token_scores[word_part]

            new_beams.append(
                (
                    new_text,
                    "",
                    word_part,
                    last_char,
                    frame_list,
                    frames,
                    logit_score,
                    logit_score + lm_score,
                    idx_history,
                )
            )
        return new_beams

    language_model = model._language_model
    if lm_start_state is None and language_model is not None:
        cached_lm_scores: Dict[str, Tuple[float, float, LMState]] = {
            "": (0.0, 0.0, language_model.get_start_state())
        }
    else:
        cached_lm_scores = {"": (0.0, 0.0, lm_start_state)}
    cached_p_lm_scores: Dict[str, float] = {}
    # start with single beam to expand on
    beams = [EMPTY_START_BEAM]

    for frame_idx, logit_col in enumerate(logits):
        max_idx = logit_col.argmax()
        idx_list = set(np.where(logit_col >= token_min_logp)[0]) | {max_idx}
        new_beams = get_new_beams(
            model,
            beams,
            idx_list,
            frame_idx,
            logit_col,
        )
        # lm scoring and beam pruning
        scored_beams = get_lm_beams(
            model,
            new_beams,
            hotword_scorer,
            cached_lm_scores,
            cached_p_lm_scores,
        )

        # remove beam outliers
        max_score = max([b[-2] for b in scored_beams])
        scored_beams = [b for b in scored_beams if b[-2] >= max_score + beam_prune_logp]
        trimmed_beams = _sort_and_trim_beams(scored_beams, beam_width)
        beams = [b[:-2] + (b[-1], ) for b in trimmed_beams]

    new_beams = []
    for text, _, word_part, _, frame_list, frames, logit_score, idx_history in beams:
        new_token_times = frame_list if word_part == "" else frame_list + [frames]
        new_beams.append((text, word_part, "", None, new_token_times, (-1, -1), logit_score, idx_history))
    new_beams = _merge_beams(new_beams)
    scored_beams = get_lm_beams(
        model,
        new_beams,
        hotword_scorer,
        cached_lm_scores,
        cached_p_lm_scores,
        is_eos=True,
    )
    scored_beams = [b[:-2] + (b[-1], ) for b in scored_beams]
    scored_beams = _merge_beams(scored_beams)

    # remove beam outliers
    max_score = max([b[-2] for b in beams])
    scored_beams = [b for b in beams if b[-2] >= max_score + beam_prune_logp]
    trimmed_beams = _sort_and_trim_beams(scored_beams, beam_width)

    # remove unnecessary information from beams
    output_beams = [
        torch.tensor(idx_history)
        for _, _, _, _, _, _, _, idx_history in trimmed_beams
    ]
    return output_beams


def decode_attn(model, enc_states, wav_len, beam_width):
    def inflate_tensor(tensor, times, dim):
        return torch.repeat_interleave(tensor, times, dim=dim)


    def mask_by_condition(tensor, cond, fill_value):
        tensor = torch.where(
            cond, tensor, torch.Tensor([fill_value]).to(tensor.device)
        )
        return tensor

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        """Performs a step in the implemented beamsearcher."""
        hs, c = memory
        e = self.emb(inp_tokens)
        dec_out, hs, c, w = self.dec.forward_step(
            e, hs, c, enc_states, enc_lens
        )
        log_probs = self.softmax(self.fc(dec_out) / self.temperature)
        if self.dec.attn_type == "multiheadlocation":
            w = torch.mean(w, dim=1)
        return log_probs, (hs, c), w

    enc_lens = torch.round(enc_states.shape[1] * wav_len).int()
    device = enc_states.device
    batch_size = enc_states.shape[0]

    memory = model.reset_mem(batch_size * beam_width, device=device)

    if model.lm_weight > 0:
        lm_memory = model.reset_lm_mem(batch_size * beam_width, device)

    if model.ctc_weight > 0:
        # (batch_size * beam_size, L, vocab_size)
        ctc_outputs = model.ctc_forward_step(enc_states)
        ctc_scorer = CTCPrefixScorer(
            ctc_outputs,
            enc_lens,
            batch_size,
            beam_width,
            model.blank_index,
            model.eos_index,
            model.ctc_window_size,
        )
        ctc_memory = None

    # Inflate the enc_states and enc_len by beam_size times
    enc_states = inflate_tensor(enc_states, times=beam_width, dim=0)
    enc_lens = inflate_tensor(enc_lens, times=beam_width, dim=0)

    # Using bos as the first input
    inp_tokens = (
        torch.zeros(batch_size * beam_width, device=device)
        .fill_(model.bos_index)
        .long()
    )

    # The first index of each sentence.
    model.beam_offset = (
        torch.arange(batch_size, device=device) * beam_width
    )

    # initialize sequence scores variables.
    sequence_scores = torch.empty(
        batch_size * beam_width, device=device
    )
    sequence_scores.fill_(float("-inf"))

    # keep only the first to make sure no redundancy.
    sequence_scores.index_fill_(0, model.beam_offset, 0.0)

    # keep the hypothesis that reaches eos and their corresponding score and log_probs.
    hyps_and_scores = [[] for _ in range(batch_size)]

    # keep the sequences that still not reaches eos.
    alived_seq = torch.empty(
        batch_size * beam_width, 0, device=device
    ).long()

    # Keep the log-probabilities of alived sequences.
    alived_log_probs = torch.empty(
        batch_size * beam_width, 0, device=device
    )

    min_decode_steps = int(enc_states.shape[1] * model.min_decode_ratio)
    max_decode_steps = int(enc_states.shape[1] * model.max_decode_ratio)

    # Initialize the previous attention peak to zero
    # This variable will be used when using_max_attn_shift=True
    prev_attn_peak = torch.zeros(batch_size * beam_width, device=device)

    for t in range(max_decode_steps):
        # terminate condition
        if model._check_full_beams(hyps_and_scores, beam_width):
            break
            
        log_probs, memory, attn = forward_step(
            model, inp_tokens, memory, enc_states, enc_lens
        )
        log_probs = model.att_weight * log_probs

        # Keep the original value
        log_probs_clone = log_probs.clone().reshape(batch_size, -1)
        vocab_size = log_probs.shape[-1]

        if model.using_max_attn_shift:
            # Block the candidates that exceed the max shift
            cond, attn_peak = model._check_attn_shift(attn, prev_attn_peak)
            log_probs = mask_by_condition(
                log_probs, cond, fill_value=model.minus_inf
            )
            prev_attn_peak = attn_peak

        # Set eos to minus_inf when less than minimum steps.
        if t < min_decode_steps:
            log_probs[:, model.eos_index] = model.minus_inf

        # Set the eos prob to minus_inf when it doesn't exceed threshold.
        if model.using_eos_threshold:
            cond = model._check_eos_threshold(log_probs)
            log_probs[:, model.eos_index] = mask_by_condition(
                log_probs[:, model.eos_index],
                cond,
                fill_value=model.minus_inf,
            )

        # adding LM scores to log_prob if lm_weight > 0
        if model.lm_weight > 0:
            lm_log_probs, lm_memory = model.lm_forward_step(
                inp_tokens, lm_memory
            )
            log_probs = log_probs + model.lm_weight * lm_log_probs
        
        # adding CTC scores to log_prob if ctc_weight > 0
        if model.ctc_weight > 0:
            g = alived_seq
            # block blank token
            log_probs[:, model.blank_index] = model.minus_inf
            if model.ctc_weight != 1.0 and model.ctc_score_mode == "partial":
                # pruning vocab for ctc_scorer
                _, ctc_candidates = log_probs.topk(
                    beam_width * 2, dim=-1
                )
            else:
                ctc_candidates = None

            ctc_log_probs, ctc_memory = ctc_scorer.forward_step(
                g, ctc_memory, ctc_candidates, attn
            )
            log_probs = log_probs + model.ctc_weight * ctc_log_probs

        scores = sequence_scores.unsqueeze(1).expand(-1, vocab_size)
        scores = scores + log_probs

        # length normalization
        if model.length_normalization:
            scores = scores / (t + 1)

        scores_timestep = scores.clone()

        # keep topk beams
        scores, candidates = scores.view(batch_size, -1).topk(
            beam_width, dim=-1
        )

        # The input for the next step, also the output of current step.
        inp_tokens = (candidates % vocab_size).view(
            batch_size * beam_width
        )

        scores = scores.view(batch_size * beam_width)
        sequence_scores = scores

        # recover the length normalization
        if model.length_normalization:
            sequence_scores = sequence_scores * (t + 1)

        # The index of which beam the current top-K output came from in (t-1) timesteps.
        predecessors = (
            torch.div(candidates, vocab_size, rounding_mode="floor")
            + model.beam_offset.unsqueeze(1).expand_as(candidates)
        ).view(batch_size * beam_width)

        # Permute the memory to synchoronize with the output.
        memory = model.permute_mem(memory, index=predecessors)
        if model.lm_weight > 0:
            lm_memory = model.permute_lm_mem(lm_memory, index=predecessors)

        if model.ctc_weight > 0:
            ctc_memory = ctc_scorer.permute_mem(ctc_memory, candidates)

        # If using_max_attn_shift, then the previous attn peak has to be permuted too.
        if model.using_max_attn_shift:
            prev_attn_peak = torch.index_select(
                prev_attn_peak, dim=0, index=predecessors
            )

        # Add coverage penalty
        if model.coverage_penalty > 0:
            cur_attn = torch.index_select(attn, dim=0, index=predecessors)

            # coverage: cumulative attention probability vector
            if t == 0:
                # Init coverage
                model.coverage = cur_attn

            # the attn of transformer is [batch_size*beam_size, current_step, source_len]
            if len(cur_attn.size()) > 2:
                model.converage = torch.sum(cur_attn, dim=1)
            else:
                # Update coverage
                model.coverage = torch.index_select(
                    model.coverage, dim=0, index=predecessors
                )
                model.coverage = model.coverage + cur_attn

            # Compute coverage penalty and add it to scores
            penalty = torch.max(
                model.coverage, model.coverage.clone().fill_(0.5)
            ).sum(-1)
            penalty = penalty - model.coverage.size(-1) * 0.5
            penalty = penalty.view(batch_size * beam_width)
            penalty = (
                penalty / (t + 1) if model.length_normalization else penalty
            )
            scores = scores - penalty * model.coverage_penalty

        # Update alived_seq
        alived_seq = torch.cat(
            [
                torch.index_select(alived_seq, dim=0, index=predecessors),
                inp_tokens.unsqueeze(1),
            ],
            dim=-1,
        )

        # Takes the log-probabilities
        beam_log_probs = log_probs_clone[
            torch.arange(batch_size).unsqueeze(1), candidates
        ].reshape(batch_size * beam_width)
        alived_log_probs = torch.cat(
            [
                torch.index_select(
                    alived_log_probs, dim=0, index=predecessors
                ),
                beam_log_probs.unsqueeze(1),
            ],
            dim=-1,
        )

        is_eos = model._update_hyp_and_scores(
            inp_tokens,
            alived_seq,
            alived_log_probs,
            hyps_and_scores,
            scores,
            timesteps=t,
        )

        # Block the paths that have reached eos.
        sequence_scores.masked_fill_(is_eos, float("-inf"))

    if not model._check_full_beams(hyps_and_scores, beam_width):
        # Using all eos to fill-up the hyps.
        eos = (
            torch.zeros(batch_size * beam_width, device=device)
            .fill_(model.eos_index)
            .long()
        )
        _ = model._update_hyp_and_scores(
            eos,
            alived_seq,
            alived_log_probs,
            hyps_and_scores,
            scores,
            timesteps=max_decode_steps,
        )
    
    topk_hyps, _, _, _, = model._get_top_score_prediction(hyps_and_scores, topk=beam_width)
    pseudo_labels = list(torch.unbind(topk_hyps.squeeze(0), dim=0))
    aux_label = [torch.tensor([model.blank_index for _ in range(max_decode_steps)])]
    pseudo_labels = pad_sequence(pseudo_labels + aux_label, batch_first=True, padding_value=model.blank_index)[:beam_width, :max_decode_steps]
    return pseudo_labels


def decode_trans(model, h, encoded_lengths):
    # Initialize states
    beam = min(model.beam_size, model.vocab_size)
    beam_k = min(beam, (model.vocab_size - 1))

    blank_tensor = torch.tensor([model.blank], device=h.device, dtype=torch.long)

    # Precompute some constants for blank position
    ids = list(range(model.vocab_size + 1))
    ids.remove(model.blank)

    # Used when blank token is first vs last token
    if model.blank == 0:
        index_incr = 1
    else:
        index_incr = 0

    # Initialize zero vector states
    dec_state = model.decoder.initialize_state(h)

    # Initialize first hypothesis for the beam (blank)
    kept_hyps = [Hypothesis(score=0.0, y_sequence=[model.blank], dec_state=dec_state, timestep=[-1], length=0, token_list=[])]
    cache = {}

    cnt = 0
    token_min_logp = -5 # need to tune

    for i in range(int(encoded_lengths)):
        hi = h[:, i : i + 1, :]  # [1, 1, D]
        hyps = kept_hyps
        kept_hyps = []

        while True:
            max_hyp = max(hyps, key=lambda x: x.score)
            hyps.remove(max_hyp)

            # update decoder state and get next score
            y, state, _ = model.decoder.score_hypothesis(max_hyp, cache)  # [1, 1, D]

            # get next token
            logit = model.joint.joint(hi, y) / model.softmax_temperature
            ytu = torch.log_softmax(logit, dim=-1)  # [1, 1, 1, V + 1]
            ytu = ytu[0, 0, 0, :]  # [V + 1]

            # remove blank token before top k
            top_k = ytu[ids].topk(beam_k, dim=-1)

            if torch.max(top_k[0]) < token_min_logp:
                top_k = (torch.masked_select(top_k[0], top_k[0] == torch.max(top_k[0])), torch.masked_select(top_k[1], top_k[0] == torch.max(top_k[0])))
            else:
                top_k = (torch.masked_select(top_k[0], top_k[0].ge(token_min_logp)), torch.masked_select(top_k[1], top_k[0].ge(token_min_logp)))

            # Two possible steps - blank token or non-blank token predicted
            ytu = (
                torch.cat((top_k[0], ytu[model.blank].unsqueeze(0))),
                torch.cat((top_k[1] + index_incr, blank_tensor)),
            )

            # for each possible step
            for logp, k in zip(*ytu):
                cnt += 1

                # construct hypothesis for step
                new_hyp = Hypothesis(
                    score=(max_hyp.score + float(logp)),
                    y_sequence=max_hyp.y_sequence[:],
                    dec_state=max_hyp.dec_state,
                    lm_state=max_hyp.lm_state,
                    timestep=max_hyp.timestep[:],
                    length=encoded_lengths,
                    token_list=max_hyp.token_list+[k],
                )

                # if current token is blank, dont update sequence, just store the current hypothesis
                if k == model.blank:
                    kept_hyps.append(new_hyp)
                else:
                    # if non-blank token was predicted, update state and sequence and then search more hypothesis
                    new_hyp.dec_state = state
                    new_hyp.y_sequence.append(int(k))
                    new_hyp.timestep.append(i)
                    hyps.append(new_hyp)

            # keep those hypothesis that have scores greater than next search generation
            hyps_max = float(max(hyps, key=lambda x: x.score).score)
            kept_most_prob = sorted([hyp for hyp in kept_hyps if hyp.score > hyps_max], key=lambda x: x.score,)

            # If enough hypothesis have scores greater than next search generation, stop beam search.
            if len(kept_most_prob) >= beam:
                kept_hyps = kept_most_prob
                break

    pseudo_labels = [torch.tensor(hyp.token_list) for hyp in model.sort_nbest(kept_hyps)]
    aux_label = [torch.tensor([model.blank for _ in range(int(encoded_lengths))])]
    pseudo_labels = pad_sequence(pseudo_labels + aux_label, batch_first=True, padding_value=model.blank)[:model.beam_size]
    return [pseudo_label for pseudo_label in pseudo_labels]