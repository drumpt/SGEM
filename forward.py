import time
import math
import logging
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple, Union
import heapq
from copy import copy

import numpy as np  # type: ignore
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.common.parts.rnn import label_collate
from audio_augmentations import *

from pyctcdecode.alphabet import BPE_TOKEN, Alphabet, verify_alphabet_coverage
from pyctcdecode.constants import (
    DEFAULT_ALPHA,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_BETA,
    DEFAULT_HOTWORD_WEIGHT,
    DEFAULT_MIN_TOKEN_LOGP,
    DEFAULT_PRUNE_BEAMS,
    DEFAULT_PRUNE_LOGP,
    DEFAULT_SCORE_LM_BOUNDARY,
    DEFAULT_UNK_LOGP_OFFSET,
    MIN_TOKEN_CLIP_P,
)
from pyctcdecode.language_model import HotwordScorer

logger = logging.getLogger(__name__)

try:
    import kenlm  # type: ignore
except ImportError:
    logger.warning(
        "kenlm python bindings are not installed. Most likely you want to install it using: "
        "pip install https://github.com/kpu/kenlm/archive/master.zip"
    )


Frames = Tuple[int, int]
WordFrames = Tuple[str, Frames]
# all the beam information we need to keep track of during decoding
# text, next_word, partial_word, last_char, text_frames, part_frames, logit_score
# Beam = Tuple[str, str, str, Optional[str], List[Frames], Frames, float]
# same as BEAMS but with current lm score that will be discarded again after sorting
LMBeam = Tuple[str, str, str, Optional[str], List[Frames], Frames, float, float]
# lm state supports single and multi language model
LMState = Optional[Union["kenlm.State", List["kenlm.State"]]]
# for output beams we return the text, the scores, the lm state and the word frame indices
# text, last_lm_state, text_frames, logit_score, lm_score
OutputBeam = Tuple[str, LMState, List[WordFrames], float, float]
# for multiprocessing we need to remove kenlm state since it can't be pickled
OutputBeamMPSafe = Tuple[str, List[WordFrames], float, float]

# constants
NULL_FRAMES: Frames = (-1, -1)  # placeholder that gets replaced with positive integer frame indices
EMPTY_START_BEAM = ("", "", "", None, [], NULL_FRAMES, 0.0, "")


from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses, is_prefix, select_k_expansions


def forward_attn(args, model, greedy_searcher, wavs, gt_wavs=None):
    log_probs_lst = []

    # current = time.time()

    enc_states = model.encode_batch(wavs, wav_lens=torch.ones(len(wavs)).to(args.device))
    enc_lens = torch.tensor([enc_states.shape[1]]).to(args.device)

    # print(f"forward encode: {time.time() - current}")
    # current = time.time()

    device = enc_states.device
    batch_size = enc_states.shape[0]
    memory = greedy_searcher.reset_mem(batch_size, device=device)

    inp_tokens = (enc_states.new_zeros(batch_size).fill_(greedy_searcher.bos_index).long())
    max_decode_steps = int(enc_states.shape[1] * greedy_searcher.max_decode_ratio)

    if gt_wavs == None:
        for _ in range(max_decode_steps):
            log_probs, memory, _ = greedy_searcher.forward_step(
                inp_tokens, memory, enc_states, enc_lens
            )
            log_probs_lst.append(log_probs)
            inp_tokens = log_probs.argmax(dim=-1)
    else:
        with torch.no_grad():
            gt_enc_states = model.encode_batch(gt_wavs, wav_lens=torch.ones(len(gt_wavs)).to(args.device))
            gt_enc_lens = torch.tensor([gt_enc_states.shape[1]]).to(args.device)

            gt_memory = greedy_searcher.reset_mem(batch_size, device=device)
            gt_inp_tokens = (gt_enc_states.new_zeros(batch_size).fill_(greedy_searcher.bos_index).long())
        for _ in range(max_decode_steps):
            log_probs, memory, _ = greedy_searcher.forward_step(
                gt_inp_tokens, memory, enc_states, enc_lens
            )

            with torch.no_grad():
                gt_log_probs, gt_memory, _ = greedy_searcher.forward_step(
                    gt_inp_tokens, gt_memory, gt_enc_states, gt_enc_lens
                )
                gt_inp_tokens = gt_log_probs.argmax(dim=-1)

            log_probs_lst.append(log_probs)

    # print(f"forward decode: {time.time() - current}")

    return log_probs_lst


def forward_trans(args, model, wavs, lens, gt_wavs=None):
    log_probs_lst = []

    if gt_wavs == None:
        # current = time.time()

        encoder_output, encoded_lengths = model(input_signal=wavs, input_signal_length=lens)
        encoder_output = encoder_output.transpose(1, 2)
        logitlen = encoded_lengths

        # print(f"forward encode : {time.time() - current}")
        # current = time.time()

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
                log_probs_lst.append(logp)

                if logp.dtype != torch.float32:
                    logp = logp.float()

                v, k = logp.max(1)
                del g

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
    else:
        encoder_output, encoded_lengths = model(input_signal=wavs, input_signal_length=lens)
        encoder_output = encoder_output.transpose(1, 2)
        logitlen = encoded_lengths

        # teacher-forcing
        gt_encoder_output, _ = model(input_signal=gt_wavs, input_signal_length=lens)
        gt_encoder_output = gt_encoder_output.transpose(1, 2)

        inseq = encoder_output  # [B, T, D]
        x, out_len, device = inseq, logitlen, inseq.device
        batchsize = x.shape[0]
        hypotheses = [rnnt_utils.Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batchsize)]
        hidden = None

        # teacher-forcing
        gt_x = gt_encoder_output
        gt_hypotheses = [rnnt_utils.Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batchsize)]
        gt_hidden = None

        if model.decoding.decoding.preserve_alignments:
            for hyp in hypotheses:
                hyp.alignments = [[]]

        last_label = torch.full([batchsize, 1], fill_value=model.decoding.decoding._blank_index, dtype=torch.long, device=device)
        blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)

        # teacher-forcing
        gt_last_label = torch.full([batchsize, 1], fill_value=model.decoding.decoding._blank_index, dtype=torch.long, device=device)
        gt_blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)

        batchsize = x.shape[0]

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
                    in_label = gt_last_label
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
                log_probs_lst.append(logp)

                if logp.dtype != torch.float32:
                    logp = logp.float()

                v, k = logp.max(1)

                k_is_blank = k == model.decoding.decoding._blank_index

                blank_mask.bitwise_or_(k_is_blank)

                if model.decoding.decoding.preserve_alignments:
                    logp_vals = logp.to('cpu')
                    logp_ids = logp_vals.max(1)[1]
                    for batch_idx in range(batchsize):
                        if time_idx < out_len[batch_idx]:
                            hypotheses[batch_idx].alignments[-2].append(
                                (logp_vals[batch_idx], logp_ids[batch_idx])
                            )

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

            gt_f = gt_x.narrow(dim=1, start=time_idx, length=1)  # [B, 1, D]

            gt_not_blank = True
            gt_symbols_added = 0

            gt_blank_mask.mul_(False)
            gt_blank_mask = time_idx >= out_len

            while gt_not_blank and (model.decoding.decoding.max_symbols is None or gt_symbols_added < model.decoding.decoding.max_symbols):
                if time_idx == 0 and gt_symbols_added == 0 and gt_hidden is None:
                    gt_in_label = model.decoding.decoding._SOS
                else:
                    gt_in_label = gt_last_label
                if isinstance(gt_in_label, torch.Tensor) and gt_in_label.dtype != torch.long:
                    gt_in_label = gt_in_label.long()
                    gt_g, gt_hidden_prime = model.decoding.decoding.decoder.predict(None, gt_hidden, False, batchsize)
                else:
                    if gt_in_label == model.decoding.decoding._SOS:
                        gt_g, gt_hidden_prime = model.decoding.decoding.decoder.predict(None, gt_hidden, False, batchsize)
                    else:
                        gt_in_label = label_collate([[gt_in_label.cpu()]])
                        gt_g, gt_hidden_prime = model.decoding.decoding.decoder.predict(gt_in_label, gt_hidden, False, batchsize)

                gt_logp = model.decoding.decoding.joint.joint(gt_f, gt_g)
                if not gt_logp.is_cuda:
                    gt_logp = gt_logp.log_softmax(dim=len(gt_logp.shape) - 1)
                gt_logp = gt_logp[:, 0, 0, :]

                if gt_logp.dtype != torch.float32:
                    gt_logp = gt_logp.float()

                gt_v, gt_k = gt_logp.max(1)

                gt_k_is_blank = gt_k == model.decoding.decoding._blank_index

                gt_blank_mask.bitwise_or_(gt_k_is_blank)

                if model.decoding.decoding.preserve_alignments:
                    gt_logp_vals = gt_logp.to('cpu')
                    gt_logp_ids = gt_logp_vals.max(1)[1]
                    for batch_idx in range(batchsize):
                        if time_idx < out_len[batch_idx]:
                            gt_hypotheses[batch_idx].alignments[-2].append(
                                (gt_logp_vals[batch_idx], gt_logp_ids[batch_idx])
                            )

                if gt_blank_mask.all():
                    gt_not_blank = False
                    if model.decoding.decoding.preserve_alignments:
                        for batch_idx in range(batchsize):
                            if len(gt_hypotheses[batch_idx].alignments[-2]) > 0:
                                gt_hypotheses[batch_idx].alignments.append([])  # blank buffer for next timestep
                else:
                    blank_indices = (gt_blank_mask == 1).nonzero(as_tuple=False)
                    if gt_hidden is not None:
                        gt_hidden_prime = model.decoding.decoding.decoder.batch_copy_states(gt_hidden_prime, gt_hidden, blank_indices)
                    elif len(blank_indices) > 0 and gt_hidden is None:
                        gt_hidden_prime = model.decoding.decoding.decoder.batch_copy_states(gt_hidden_prime, None, blank_indices, value=0.0)
                    gt_k[blank_indices] = gt_last_label[blank_indices, 0]
                    gt_last_label = gt_k.clone().view(-1, 1)
                    gt_hidden = gt_hidden_prime
                    for kidx, ki in enumerate(gt_k):
                        if gt_blank_mask[kidx] == 0:
                            gt_hypotheses[kidx].y_sequence.append(ki)
                            gt_hypotheses[kidx].timestep.append(time_idx)
                            gt_hypotheses[kidx].score += float(gt_v[kidx])
                    gt_symbols_added += 1

    # print(f"forward decode : {time.time() - current}")

    return log_probs_lst


def decode_beams_ctc(
        model,
        logits,  # type: ignore [type-arg]
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
        for text, next_word, word_part, last_char, text_frames, part_frames, logit_score, char_history in beams:
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
                    char_history
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
                    char_history
                )
        return list(beam_dict.values())


    def _sort_and_trim_beams(beams, beam_width: int):
        """Take top N beams by score."""
        return heapq.nlargest(beam_width, beams, key=lambda x: x[-2])


    def _prune_history(beams, lm_order: int):
        """Filter out beams that are the same over max_ngram history.

        Since n-gram language models have a finite history when scoring a new token, we can use that
        fact to prune beams that only differ early on (more than n tokens in the past) and keep only the
        higher scoring ones. Note that this helps speed up the decoding process but comes at the cost of
        some amount of beam diversity. If more than the top beam is used in the output it should
        potentially be disabled.
        """
        # let's keep at least 1 word of history
        min_n_history = max(1, lm_order - 1)
        seen_hashes = set()
        filtered_beams = []
        # for each beam after this, check if we need to add it
        for (text, next_word, word_part, last_char, text_frames, part_frames, logit_score, char_history) in beams:
            # hash based on history that can still affect lm scoring going forward
            hash_idx = (tuple(text.split()[-min_n_history:]), word_part, last_char)
            if hash_idx not in seen_hashes:
                filtered_beams.append(
                    (
                        text,
                        next_word,
                        word_part,
                        last_char,
                        text_frames,
                        part_frames,
                        logit_score,
                        char_history
                    )
                )
                seen_hashes.add(hash_idx)
        return filtered_beams


    def _normalize_whitespace(text: str) -> str:
        """Efficiently normalize whitespace."""
        return " ".join(text.split())


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
            for text, next_word, word_part, last_char, frame_list, frames, logit_score, char_history in beams:
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
                        char_history
                    )
                )
            return new_beams

        new_beams = []
        for text, next_word, word_part, last_char, frame_list, frames, logit_score, char_history in beams:
            # fast token merge
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
                    char_history
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
    # bpe we can also have trailing word boundaries ▁⁇▁ so we may need to remember breaks
    force_next_break = False
    for frame_idx, logit_col in enumerate(logits):
        max_idx = logit_col.argmax()
        idx_list = set(np.where(logit_col >= token_min_logp)[0]) | {max_idx}
        new_beams = []
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
                char_history,
            ) in beams:
                # if only blank token or same token
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
                            char_history + char if char != "" else char_history + "|"
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
                            char_history + char if char != "" else char_history + "|"
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
                            char_history + char if char != "" else char_history + "|"
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
                            char_history + char if char != "" else char_history + "|"
                        )
                    )
        # lm scoring and beam pruning
        new_beams = _merge_beams(new_beams)

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
        # beam pruning by taking highest N prefixes and then filtering down
        trimmed_beams = _sort_and_trim_beams(scored_beams, beam_width)
        # prune history and remove lm score from beams
        # if prune_history:
        #     lm_order = 1 if language_model is None else language_model.order
        #     beams = _prune_history(trimmed_beams, lm_order=lm_order)
        # else:
        #     beams = trimmed_beams
        #     beams = [b[:-2] for b in trimmed_beams]
        beams = [b[:-2] + (b[-1], ) for b in trimmed_beams]

    # final lm scoring and sorting
    # print(f"beams: {beams}")

    new_beams = []
    for text, _, word_part, _, frame_list, frames, logit_score, char_history in beams:
        new_token_times = frame_list if word_part == "" else frame_list + [frames]
        new_beams.append((text, word_part, "", None, new_token_times, (-1, -1), logit_score, char_history))
    new_beams = _merge_beams(new_beams)
    scored_beams = get_lm_beams(
        model,
        new_beams,
        hotword_scorer,
        cached_lm_scores,
        cached_p_lm_scores,
        is_eos=True,
    )

    # remove beam outliers
    max_score = max([b[-2] for b in scored_beams])
    scored_beams = [b for b in scored_beams if b[-2] >= max_score + beam_prune_logp]
    trimmed_beams = _sort_and_trim_beams(scored_beams, beam_width)

    # remove unnecessary information from beams
    output_beams = [
        (
            _normalize_whitespace(text),
            cached_lm_scores[text][-2] if text in cached_lm_scores else None,
            list(zip(text.split(), text_frames)),
            logit_score,
            combined_score,  # same as logit_score if lm is missing
            char_history
        )
        for text, _, _, _, text_frames, _, logit_score, combined_score, char_history in trimmed_beams
    ]

    ### TOOD: remove (only for debugging)
    # print(f"trimmed_beams: {trimmed_beams}")
    # for text, _, _, char, text_frames, _, logit_score, combined_score in trimmed_beams:
    #     print(f"char!!!: {char}")

    return output_beams


def decode_beams_trans(
    model, h: torch.Tensor, encoded_lengths: torch.Tensor, partial_hypotheses: Optional[Hypothesis] = None
) -> List[Hypothesis]:
    """Beam search implementation.

    Args:
        x: Encoded speech features (1, T_max, D_enc)

    Returns:
        nbest_hyps: N-best decoding results
    """
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
    kept_hyps = [Hypothesis(score=0.0, y_sequence=[model.blank], dec_state=dec_state, timestep=[-1], length=0)]
    cache = {}

    if partial_hypotheses is not None:
        if len(partial_hypotheses.y_sequence) > 0:
            kept_hyps[0].y_sequence = [int(partial_hypotheses.y_sequence[-1].cpu().numpy())]
            kept_hyps[0].dec_state = partial_hypotheses.dec_state
            kept_hyps[0].dec_state = _states_to_device(kept_hyps[0].dec_state, h.device)

    if model.preserve_alignments:
        kept_hyps[0].alignments = [[]]

    for i in range(int(encoded_lengths)):
        hi = h[:, i : i + 1, :]  # [1, 1, D]
        hyps = kept_hyps
        kept_hyps = []

        while True:
            max_hyp = max(hyps, key=lambda x: x.score)
            hyps.remove(max_hyp)

            # update decoder state and get next score
            y, state, lm_tokens = model.decoder.score_hypothesis(max_hyp, cache)  # [1, 1, D]

            # get next token
            ytu = torch.log_softmax(model.joint.joint(hi, y) / model.softmax_temperature, dim=-1)  # [1, 1, 1, V + 1]
            ytu = ytu[0, 0, 0, :]  # [V + 1]

            # remove blank token before top k
            top_k = ytu[ids].topk(beam_k, dim=-1)

            # Two possible steps - blank token or non-blank token predicted
            ytu = (
                torch.cat((top_k[0], ytu[model.blank].unsqueeze(0))),
                torch.cat((top_k[1] + index_incr, blank_tensor)),
            )

            # for each possible step
            for logp, k in zip(*ytu):
                # construct hypothesis for step
                new_hyp = Hypothesis(
                    score=(max_hyp.score + float(logp)),
                    y_sequence=max_hyp.y_sequence[:],
                    dec_state=max_hyp.dec_state,
                    lm_state=max_hyp.lm_state,
                    timestep=max_hyp.timestep[:],
                    length=encoded_lengths,
                )

                if model.preserve_alignments:
                    new_hyp.alignments = copy.deepcopy(max_hyp.alignments)

                # if current token is blank, dont update sequence, just store the current hypothesis
                if k == model.blank:
                    kept_hyps.append(new_hyp)
                else:
                    # if non-blank token was predicted, update state and sequence and then search more hypothesis
                    new_hyp.dec_state = state
                    new_hyp.y_sequence.append(int(k))
                    new_hyp.timestep.append(i)

                    hyps.append(new_hyp)

                if model.preserve_alignments:
                    if k == model.blank:
                        new_hyp.alignments[-1].append(model.blank)
                    else:
                        new_hyp.alignments[-1].append(new_hyp.y_sequence[-1])

            # keep those hypothesis that have scores greater than next search generation
            hyps_max = float(max(hyps, key=lambda x: x.score).score)
            kept_most_prob = sorted([hyp for hyp in kept_hyps if hyp.score > hyps_max], key=lambda x: x.score,)

            # If enough hypothesis have scores greater than next search generation,
            # stop beam search.
            if len(kept_most_prob) >= beam:
                if model.preserve_alignments:
                    # convert Ti-th logits into a torch array
                    for kept_h in kept_most_prob:
                        kept_h.alignments.append([])  # blank buffer for next timestep

                kept_hyps = kept_most_prob
                break

    # Remove trailing empty list of alignments
    if model.preserve_alignments:
        for h in kept_hyps:
            if len(h.alignments[-1]) == 0:
                del h.alignments[-1]

    # Remove the original input label if partial hypothesis was provided
    if partial_hypotheses is not None:
        for hyp in kept_hyps:
            if hyp.y_sequence[0] == partial_hypotheses.y_sequence[-1] and len(hyp.y_sequence) > 1:
                hyp.y_sequence = hyp.y_sequence[1:]

    

    return model.sort_nbest(kept_hyps)