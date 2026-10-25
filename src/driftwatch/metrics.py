"""Distribution-distance metrics and binning helpers.

All functions operate on plain numpy arrays so they can be used on their own,
independently of :class:`driftwatch.DriftDetector`.

Binned metrics (PSI, Jensen-Shannon, total variation) expect the reference and
current distributions to be expressed over the *same* bins, as either raw
counts or proportions. The chi-square test needs raw counts.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

#: Floor applied to empty bins before taking logarithms. Without it a bin that
#: is empty in one distribution would make PSI infinite. 1e-4 means an empty
#: bin is treated as holding 0.01% of the mass, which keeps a single empty bin
#: from dominating the score while still penalising it heavily.
EPSILON = 1e-4


def _as_distribution(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("distribution must be a non-empty 1-D sequence")
    if np.any(arr < 0) or not np.all(np.isfinite(arr)):
        raise ValueError("distribution must contain finite, non-negative values")
    total = arr.sum()
    if total <= 0:
        raise ValueError("distribution has no mass (all bins are zero)")
    return arr / total


def _check_same_shape(p: np.ndarray, q: np.ndarray) -> None:
    if p.shape != q.shape:
        raise ValueError(f"distributions have different numbers of bins: {p.size} vs {q.size}")


def smooth(values: Sequence[float], eps: float = EPSILON) -> np.ndarray:
    """Normalise to proportions and floor every bin at ``eps``, then renormalise."""
    p = np.clip(_as_distribution(values), eps, None)
    return p / p.sum()


def psi(expected: Sequence[float], actual: Sequence[float], eps: float = EPSILON) -> float:
    """Population Stability Index.

    ``PSI = sum((a_i - e_i) * ln(a_i / e_i))`` over shared bins, where ``e`` is
    the reference (expected) distribution and ``a`` the current (actual) one.
    Empty bins are floored at ``eps`` so the result is always finite.
    Identical distributions give exactly 0.
    """
    e = smooth(expected, eps)
    a = smooth(actual, eps)
    _check_same_shape(e, a)
    return float(np.sum((a - e) * np.log(a / e)))


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence with log base 2, bounded to ``[0, 1]``.

    Unlike PSI it is symmetric and well defined for empty bins, so no smoothing
    is applied. Note this returns the *divergence* (the square of SciPy's
    Jensen-Shannon distance).
    """
    p_ = _as_distribution(p)
    q_ = _as_distribution(q)
    _check_same_shape(p_, q_)
    value = float(jensenshannon(p_, q_, base=2) ** 2)
    return min(max(value, 0.0), 1.0)


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    """Total variation distance ``0.5 * sum|p_i - q_i|`` in ``[0, 1]``.

    Reads as "the share of probability mass that has to move to turn one
    distribution into the other".
    """
    p_ = _as_distribution(p)
    q_ = _as_distribution(q)
    _check_same_shape(p_, q_)
    return float(0.5 * np.abs(p_ - q_).sum())


def ks_test(reference: Sequence[float], current: Sequence[float]) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns ``(statistic, p_value)``.

    NaNs are dropped. Raises ``ValueError`` if either sample is empty.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError("KS test needs at least one non-missing value in each sample")
    result = stats.ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


def chi_square_test(
    reference_counts: Sequence[float],
    current_counts: Sequence[float],
) -> Tuple[float, float]:
    """Two-sample chi-square test of homogeneity on binned counts.

    Builds a 2 x k contingency table (reference vs current) and tests whether
    both samples come from the same categorical distribution. Unlike a
    goodness-of-fit test against fixed reference proportions, this accounts for
    sampling noise in the reference data too, which keeps false alarms near
    the nominal rate. Bins that are empty in both samples are dropped; with
    fewer than two remaining bins there is nothing to test and ``(0.0, 1.0)``
    is returned. Returns ``(statistic, p_value)``.
    """
    ref = np.asarray(reference_counts, dtype=float)
    cur = np.asarray(current_counts, dtype=float)
    _check_same_shape(ref, cur)
    if ref.sum() <= 0 or cur.sum() <= 0:
        raise ValueError("chi-square test needs at least one observation in each sample")
    keep = (ref + cur) > 0
    if keep.sum() < 2:
        return 0.0, 1.0
    table = np.vstack([ref[keep], cur[keep]])
    statistic, p_value, _, _ = stats.chi2_contingency(table, correction=False)
    return float(statistic), float(p_value)


def quantile_edges(values: Sequence[float], n_bins: int) -> list:
    """Interior bin edges at the reference quantiles.

    Returns at most ``n_bins - 1`` strictly increasing edges; duplicates (which
    occur for discrete or heavily tied data) are collapsed, so low-cardinality
    columns simply get fewer bins. The outer bins are open-ended, so values
    outside the reference range still land in a bin.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return []
    probs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.quantile(arr, probs)).tolist()


def bin_counts(values: Sequence[float], edges: Sequence[float]) -> np.ndarray:
    """Count values into bins ``(-inf, e0), [e0, e1), ..., [e_last, inf)`` plus a final missing bin.

    The returned array has ``len(edges) + 2`` entries; the last one counts NaNs.
    """
    arr = np.asarray(values, dtype=float)
    missing = np.isnan(arr)
    idx = np.searchsorted(np.asarray(edges, dtype=float), arr[~missing], side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return np.append(counts, float(missing.sum()))
