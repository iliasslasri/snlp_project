import torch
import random
import numpy as np
import os
import logging
from typing import List


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, "train.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(seq: List[int]) -> List[int]:
    """Remove consecutive repeated units, e.g. [1,1,2,3,3] -> [1,2,3].
    This is the standard post-processing step in GSLM (Lakhotia et al., 2021).
    """
    if not seq:
        return []
    deduped = [seq[0]]
    for token in seq[1:]:
        if token != deduped[-1]:
            deduped.append(token)
    return deduped


# ---------------------------------------------------------------------------
# Levenshtein distance  (Appendix A of the paper)
# ---------------------------------------------------------------------------

def levenshtein(x: List[int], y: List[int]) -> int:
    """Compute the Levenshtein (edit) distance between two integer sequences.

    Uses a standard DP table — O(|x| * |y|) time, O(min(|x|,|y|)) space.
    This matches the recursive definition given in Appendix A of the paper:

        Lev(x, y) = |x|               if |y| = 0
                    |y|               if |x| = 0
                    1 + min(Lev(tail(x), y),
                            Lev(x, tail(y)),
                            Lev(tail(x), tail(y)))   otherwise
    """
    nx, ny = len(x), len(y)
    if nx == 0:
        return ny
    if ny == 0:
        return nx

    # Keep only two rows to save memory
    prev = list(range(ny + 1))
    curr = [0] * (ny + 1)

    for i in range(1, nx + 1):
        curr[0] = i
        for j in range(1, ny + 1):
            substitution_cost = 0 if x[i - 1] == y[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,                        # deletion
                curr[j - 1] + 1,                    # insertion
                prev[j - 1] + substitution_cost,    # substitution
            )
        prev, curr = curr, prev

    return prev[ny]


# ---------------------------------------------------------------------------
# Unit Edit Distance  (Definition 3.1 of the paper)
# ---------------------------------------------------------------------------

def unit_edit_distance(
    original_units: List[List[int]],
    augmented_units: List[List[int]],
) -> float:
    """Compute the mean Unit Edit Distance (UED) over a batch.

    For each sample x in the evaluation set D:

        UED_D(E, f, g) = sum_{x in D}  (1 / T'_x) * LEV(dedup(E∘f)(x), dedup(E∘f∘g)(x))

    Both sequences are deduplicated before computing the distance so that
    time-stretching artefacts (repeated frames) do not inflate the score.

    Args:
        original_units:  List of unit sequences from the clean signal.
                         Each element is a list of integer unit ids.
        augmented_units: List of unit sequences from the augmented signal.
                         Same length as original_units.

    Returns:
        Mean UED score multiplied by 100 (as reported in the paper).
    """
    assert len(original_units) == len(augmented_units), (
        "original_units and augmented_units must have the same length"
    )

    total = 0.0
    for orig, aug in zip(original_units, augmented_units):
        orig_dedup = deduplicate(orig)
        aug_dedup  = deduplicate(aug)

        T_prime = len(orig_dedup)
        if T_prime == 0:
            continue  # skip silent / empty samples

        dist = levenshtein(orig_dedup, aug_dedup)
        # Normalise by the length of the original deduplicated sequence
        total += dist / T_prime

    mean_ued = total / len(original_units) if original_units else 0.0
    # Multiply by 100 for readability, consistent with the paper's tables
    return mean_ued * 100


# ---------------------------------------------------------------------------
# Convenience wrapper for tensors (from model outputs)
# ---------------------------------------------------------------------------

def compute_ued_from_tensors(
    original_units: torch.Tensor,
    augmented_units: torch.Tensor,
    lengths_orig: torch.Tensor = None,
    lengths_aug: torch.Tensor = None,
) -> float:
    """Same as unit_edit_distance but accepts 2-D padded tensors.

    Args:
        original_units:  [B, T] integer tensor of unit ids (padded with 0).
        augmented_units: [B, T] integer tensor of unit ids (padded with 0).
        lengths_orig:    [B] tensor with the valid length of each sequence.
        lengths_aug:     [B] tensor with the valid length of each sequence.
    """
    B = original_units.shape[0]

    orig_list = []
    aug_list  = []

    for i in range(B):
        lo = lengths_orig[i].item() if lengths_orig is not None else original_units.shape[1]
        la = lengths_aug[i].item()  if lengths_aug  is not None else augmented_units.shape[1]
        orig_list.append(original_units[i, :lo].tolist())
        aug_list.append(augmented_units[i, :la].tolist())

    return unit_edit_distance(orig_list, aug_list)