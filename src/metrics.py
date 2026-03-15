"""Core evaluation metrics for GSLM (Generative Spoken Language Modeling).

Implements:
- Unit Edit Distance (UED) – robustness metric from the paper
- sWUGGY scoring – lexical evaluation via LM log-likelihood
- sBLIMP scoring – syntactic evaluation via LM log-likelihood
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, List, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def deduplicate(units: Union[List[int], torch.Tensor]) -> List[int]:
    """Remove consecutive duplicate units (unique_consecutive).

    Example: [1, 1, 2, 2, 1] -> [1, 2, 1]
    """
    if isinstance(units, torch.Tensor):
        units = units.tolist()
    if not units:
        return []
    deduped: List[int] = [units[0]]
    for u in units[1:]:
        if u != deduped[-1]:
            deduped.append(u)
    return deduped


def levenshtein_distance(seq1: List[int], seq2: List[int]) -> int:
    """Compute Levenshtein edit distance between two integer sequences.

    Uses the ``editdistance`` package (C implementation, ~20× faster than
    pure Python) when available, with a pure-Python two-row DP fallback.

    Time: O(m*n), Space: O(n).
    """
    try:
        import editdistance  # pip install editdistance

        return int(editdistance.eval(seq1, seq2))
    except ImportError:
        pass

    # Pure-Python fallback
    m, n = len(seq1), len(seq2)
    if m == 0:
        return n
    if n == 0:
        return m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


# ---------------------------------------------------------------------------
# Unit Edit Distance (UED)
# ---------------------------------------------------------------------------


def unit_edit_distance(
    clean_units: Union[List[int], torch.Tensor],
    aug_units: Union[List[int], torch.Tensor],
) -> float:
    """Compute Unit Edit Distance (UED) between a clean and augmented unit sequence.

    As defined in "Augmentation Invariant Discrete Representation for GSLM":

        1. Deduplicate both sequences (remove consecutive repetitions).
        2. Compute Levenshtein distance between the deduplicated sequences.
        3. Divide by the length of the (deduplicated) clean sequence T_x'.
        4. Multiply by 100 to match the paper's scale.

    Args:
        clean_units: Discrete unit sequence for the original (clean) audio.
        aug_units:   Discrete unit sequence for the augmented audio.

    Returns:
        UED score (percentage, lower is better).
    """
    clean_deduped = deduplicate(clean_units)
    aug_deduped = deduplicate(aug_units)

    t_x = len(clean_deduped)
    if t_x == 0:
        return 0.0

    dist = levenshtein_distance(clean_deduped, aug_deduped)
    return (dist / t_x) * 100.0


def batch_unit_edit_distance(
    clean_batch: List[Union[List[int], torch.Tensor]],
    aug_batch: List[Union[List[int], torch.Tensor]],
) -> Tuple[float, List[float]]:
    """Compute UED for a list of (clean, augmented) sequence pairs.

    Returns:
        (mean_ued, per_sample_ueds)
    """
    scores = [unit_edit_distance(c, a) for c, a in zip(clean_batch, aug_batch)]
    return float(np.mean(scores)), scores


# ---------------------------------------------------------------------------
# Language-model scoring
# ---------------------------------------------------------------------------


def sequence_log_likelihood(
    lm_score_fn: Callable[[List[int]], float],
    sequence: List[int],
) -> float:
    """Wrapper: compute log-likelihood of a unit sequence under an LM.

    Args:
        lm_score_fn: Callable that maps a unit sequence to its log-likelihood.
        sequence:    List of integer unit indices.

    Returns:
        Total log-likelihood (scalar).
    """
    return lm_score_fn(sequence)


def swuggy_score(
    lm_score_fn: Callable[[List[int]], float],
    real_sequences: List[List[int]],
    pseudo_sequences: List[List[int]],
) -> float:
    """Compute sWUGGY accuracy (lexical evaluation).

    For each (real_word, pseudo_word) pair, the LM should assign a higher
    log-probability to the real word.

    Args:
        lm_score_fn:      Callable(sequence) -> log_likelihood (scalar).
        real_sequences:   Unit sequences corresponding to real words.
        pseudo_sequences: Unit sequences corresponding to pseudo-words (non-words).

    Returns:
        sWUGGY accuracy (%, higher is better).
    """
    if not real_sequences:
        return 0.0
    correct = sum(
        1
        for real, pseudo in zip(real_sequences, pseudo_sequences)
        if lm_score_fn(real) > lm_score_fn(pseudo)
    )
    return 100.0 * correct / len(real_sequences)


def sblim_score(
    lm_score_fn: Callable[[List[int]], float],
    good_sequences: List[List[int]],
    bad_sequences: List[List[int]],
) -> float:
    """Compute sBLIMP accuracy (syntactic evaluation).

    For each (grammatical, ungrammatical) sentence pair, the LM should assign
    a higher log-probability to the grammatical sentence.

    Args:
        lm_score_fn:    Callable(sequence) -> log_likelihood (scalar).
        good_sequences: Unit sequences for syntactically correct sentences.
        bad_sequences:  Unit sequences for syntactically incorrect sentences.

    Returns:
        sBLIMP accuracy (%, higher is better).
    """
    if not good_sequences:
        return 0.0
    correct = sum(
        1
        for good, bad in zip(good_sequences, bad_sequences)
        if lm_score_fn(good) > lm_score_fn(bad)
    )
    return 100.0 * correct / len(good_sequences)


# ---------------------------------------------------------------------------
# Simple n-gram Language Model
# ---------------------------------------------------------------------------


class NgramLM:
    """Simple n-gram language model with Laplace smoothing.

    Used as a fallback when no external LM is available for sWUGGY / sBLIMP.
    Trains on lists of integer unit sequences.
    """

    def __init__(self, n: int = 3, vocab_size: int = 500, smoothing: float = 1.0):
        self.n = n
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        # context tuple -> {next_token -> count}
        self._ngram: dict = defaultdict(lambda: defaultdict(int))
        # context tuple -> total count
        self._ctx: dict = defaultdict(int)

    # ------------------------------------------------------------------
    def train(self, sequences: List[List[int]]) -> None:
        """Train on a list of unit sequences."""
        self._ngram = defaultdict(lambda: defaultdict(int))
        self._ctx = defaultdict(int)
        pad = [None] * (self.n - 1)
        for seq in sequences:
            padded = pad + seq
            for i in range(self.n - 1, len(padded)):
                ctx = tuple(padded[i - self.n + 1 : i])
                tok = padded[i]
                self._ngram[ctx][tok] += 1
                self._ctx[ctx] += 1

    def log_prob(self, sequence: List[int]) -> float:
        """Compute log-probability of a unit sequence (sum of token log-probs)."""
        pad = [None] * (self.n - 1)
        padded = pad + sequence
        total = 0.0
        for i in range(self.n - 1, len(padded)):
            ctx = tuple(padded[i - self.n + 1 : i])
            tok = padded[i]
            count = self._ngram.get(ctx, {}).get(tok, 0)
            ctx_count = self._ctx.get(ctx, 0)
            prob = (count + self.smoothing) / (
                ctx_count + self.smoothing * self.vocab_size
            )
            total += math.log(prob)
        return total

    def save(self, path: str) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "n": self.n,
                    "vocab_size": self.vocab_size,
                    "smoothing": self.smoothing,
                    "ngram": dict(self._ngram),
                    "ctx": dict(self._ctx),
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "NgramLM":
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        lm = cls(n=data["n"], vocab_size=data["vocab_size"], smoothing=data["smoothing"])
        lm._ngram = defaultdict(lambda: defaultdict(int), data["ngram"])
        lm._ctx = defaultdict(int, data["ctx"])
        return lm
