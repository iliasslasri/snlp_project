"""
metrics.py — Robustness metrics for discrete speech representations.

Implements the Unit Edit Distance (UED) from Definition 3.1 of:
  "Augmentation Invariant Discrete Representation for
   Generative Spoken Language Modeling" (Gat et al., IWSLT 2023)
"""

from typing import List


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(seq: List[int]) -> List[int]:
    """Remove consecutive repeated units.

    e.g. [1, 1, 2, 3, 3, 3] -> [1, 2, 3]

    This is the standard post-processing step in GSLM (Lakhotia et al., 2021):
    a pseudo-text like [10, 11, 11, 21, 32, 32] becomes [10, 11, 21, 32].
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

    Matches the recursive definition in Appendix A:
        Lev(x, y) = |x|                          if |y| = 0
                    |y|                          if |x| = 0
                    1 + min(Lev(tail(x), y),
                            Lev(x, tail(y)),
                            Lev(tail(x), tail(y)))  otherwise

    Uses an iterative DP table — O(|x|·|y|) time, O(min(|x|,|y|)) space.
    """
    nx, ny = len(x), len(y)
    if nx == 0:
        return ny
    if ny == 0:
        return nx

    prev = list(range(ny + 1))
    curr = [0] * (ny + 1)

    for i in range(1, nx + 1):
        curr[0] = i
        for j in range(1, ny + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev

    return prev[ny]


# ---------------------------------------------------------------------------
# Unit Edit Distance  (Definition 3.1)
# ---------------------------------------------------------------------------

def unit_edit_distance(
    original_units: List[List[int]],
    augmented_units: List[List[int]],
) -> float:
    """Compute the mean Unit Edit Distance (UED) over a set of samples.

    UED_D(E, f, g) = (1/|D|) * sum_{x in D} LEV(dedup(E∘f)(x), dedup(E∘f∘g)(x)) / T'_x

    Both sequences are deduplicated before computing the distance so that
    time-stretching artefacts (repeated frames) do not inflate the score.
    The result is multiplied by 100 for readability, consistent with the
    paper's tables.

    Args:
        original_units:  List of unit sequences from the clean signal.
        augmented_units: List of unit sequences from the augmented signal.

    Returns:
        Mean UED × 100.
    """
    assert len(original_units) == len(augmented_units), (
        "original_units and augmented_units must have the same length"
    )

    total = 0.0
    n = 0
    for orig, aug in zip(original_units, augmented_units):
        orig_dedup = deduplicate(orig)
        aug_dedup  = deduplicate(aug)

        T_prime = len(orig_dedup)
        if T_prime == 0:
            continue

        total += levenshtein(orig_dedup, aug_dedup) / T_prime
        n += 1

    return (total / n * 100) if n > 0 else 0.0