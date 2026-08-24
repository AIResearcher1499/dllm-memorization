"""Morris et al. (2505.24832) capacity arithmetic on uniform random data.

On uniform data the reference code length is analytic (S * log2 V bits per
sequence), so unintended memorization reduces to
    mem_U(x) = max(0, ref_bits - nll_bits(x))
where nll_bits is the model's total negative log-likelihood of x in bits
(exact NLL for AR / chain-rule MDM; ELBO for the achievable-compression bound —
the caller labels which estimator produced it, see prereg §3).
"""

import math
from typing import Iterable, Sequence, Tuple

VOCAB = 2048
SEQ_LEN = 64
REF_BITS_PER_SEQ = SEQ_LEN * math.log2(VOCAB)  # 704.0


def memorized_bits(nll_bits: float, ref_bits: float = REF_BITS_PER_SEQ) -> float:
    """Unintended memorization of one sequence, in bits (never negative)."""
    if nll_bits < 0:
        raise ValueError("NLL in bits cannot be negative")
    return max(0.0, ref_bits - nll_bits)


def total_memorized_bits(nll_bits_per_seq: Iterable[float]) -> float:
    return sum(memorized_bits(b) for b in nll_bits_per_seq)


def plateau_alpha(points: Sequence[Tuple[int, float]]) -> float:
    """Least-squares slope of capacity plateau (bits) vs param count.

    points: (n_params, plateau_bits) per model size; requires >= 2 sizes.
    Returns alpha in bits/param. Which param count enters (raw vs
    non-embedding) is fixed in data/config_lock.json per prereg §2.
    """
    if len(points) < 2:
        raise ValueError("need at least two model sizes")
    n = len(points)
    sx = sum(p for p, _ in points)
    sy = sum(b for _, b in points)
    sxx = sum(p * p for p, _ in points)
    sxy = sum(p * b for p, b in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        raise ValueError("degenerate param counts")
    return (n * sxy - sx * sy) / denom


def has_plateau(mem_by_dataset_size: Sequence[Tuple[int, float]], tol: float = 0.10) -> bool:
    """Prereg §2: plateau = memorization within tol of its max at >= 2
    consecutive |D| points (input sorted by |D| ascending)."""
    if len(mem_by_dataset_size) < 2:
        return False
    peak = max(b for _, b in mem_by_dataset_size)
    if peak <= 0:
        return False
    near = [b >= (1 - tol) * peak for _, b in mem_by_dataset_size]
    return any(a and b for a, b in zip(near, near[1:]))
