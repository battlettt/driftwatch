"""Multivariate drift: compare correlation matrices of numeric columns.

Per-column checks miss a change in how features move *together*. If income
and loan amount used to be strongly correlated and no longer are, each
marginal distribution can look unchanged while the model, which learned the
joint pattern, quietly goes wrong. This module stores the reference
correlation matrix at fit time and, for each new batch, reports the column
pairs whose correlation changed most.

Correlations use pairwise-complete observations (rows where both columns are
present). A pair needs at least :data:`MIN_PAIR_ROWS` such rows in a batch;
otherwise its correlation is reported as undefined rather than estimated from
too few points. A pair is also undefined when either column is constant.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

METHODS = ("pearson", "spearman")
#: Minimum number of rows with both values present before a pair's correlation is computed.
MIN_PAIR_ROWS = 30


@dataclass
class CorrelationProfile:
    """Reference correlation matrix of the numeric columns, stored with the detector.

    ``matrix[i][j]`` is the correlation of ``columns[i]`` and ``columns[j]``,
    or NaN when undefined.
    """

    columns: List[str]
    matrix: List[List[float]]
    method: str = "pearson"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorrelationProfile":
        matrix = [[float("nan") if v is None else float(v) for v in row] for row in data["matrix"]]
        return cls(columns=list(data["columns"]), matrix=matrix, method=data.get("method", "pearson"))


@dataclass
class CorrelationShift:
    """How the correlation between two columns changed."""

    column_a: str
    column_b: str
    reference: float
    current: float
    change: float
    abs_change: float
    exceeds_threshold: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CorrelationReport:
    """Result of comparing a batch's correlation matrix against the reference."""

    method: str
    threshold: float
    columns: List[str]
    pairs: List[CorrelationShift]
    undefined_pairs: List[List[str]] = field(default_factory=list)
    top_k: int = 5

    @property
    def drifted(self) -> bool:
        """True when any pair's absolute correlation change reaches the threshold."""
        return any(p.exceeds_threshold for p in self.pairs)

    @property
    def max_abs_change(self) -> float:
        return max((p.abs_change for p in self.pairs), default=0.0)

    @property
    def flagged_pairs(self) -> List[CorrelationShift]:
        return [p for p in self.pairs if p.exceeds_threshold]

    @property
    def top_shifts(self) -> List[CorrelationShift]:
        """The ``top_k`` pairs with the largest absolute change, plus any other flagged pairs."""
        return self.pairs[: max(self.top_k, len(self.flagged_pairs))]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "drifted": self.drifted,
            "max_abs_change": self.max_abs_change,
            "columns": self.columns,
            "n_pairs": len(self.pairs),
            "flagged_pairs": [[p.column_a, p.column_b] for p in self.flagged_pairs],
            "top_shifts": [p.to_dict() for p in self.top_shifts],
            "pairs": [p.to_dict() for p in self.pairs],
            "undefined_pairs": self.undefined_pairs,
        }


def correlation_matrix(df: pd.DataFrame, columns: Sequence[str], method: str = "pearson") -> pd.DataFrame:
    """Pairwise-complete correlation matrix of ``columns`` (non-numeric values become NaN)."""
    if method not in METHODS:
        raise ValueError(f"correlation method must be one of {METHODS}, got {method!r}")
    numeric = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in columns}, columns=list(columns))
    numeric = numeric.astype(float)
    return numeric.corr(method=method, min_periods=MIN_PAIR_ROWS)


def fit_correlation(df: pd.DataFrame, columns: Sequence[str], method: str = "pearson") -> Optional[CorrelationProfile]:
    """Reference profile for ``columns``, or ``None`` when there are fewer than two of them."""
    columns = list(columns)
    if len(columns) < 2:
        return None
    corr = correlation_matrix(df, columns, method)
    return CorrelationProfile(
        columns=columns,
        matrix=[[float(v) for v in row] for row in corr.to_numpy()],
        method=method,
    )


def compare_correlations(
    profile: CorrelationProfile,
    current: pd.DataFrame,
    threshold: float,
    top_k: int = 5,
) -> Optional[CorrelationReport]:
    """Compare ``current`` against the reference correlations.

    Columns missing from ``current`` are skipped. Returns ``None`` when fewer
    than two of the profiled columns are present.
    """
    columns = [c for c in profile.columns if c in current.columns]
    if len(columns) < 2:
        return None
    index = {c: i for i, c in enumerate(profile.columns)}
    cur = correlation_matrix(current, columns, profile.method).to_numpy()

    pairs: List[CorrelationShift] = []
    undefined: List[List[str]] = []
    for i, a in enumerate(columns):
        for j in range(i + 1, len(columns)):
            b = columns[j]
            ref_value = profile.matrix[index[a]][index[b]]
            cur_value = float(cur[i, j])
            if math.isnan(ref_value) or math.isnan(cur_value):
                undefined.append([a, b])
                continue
            change = cur_value - ref_value
            abs_change = abs(change)
            pairs.append(
                CorrelationShift(
                    column_a=a,
                    column_b=b,
                    reference=ref_value,
                    current=cur_value,
                    change=change,
                    abs_change=abs_change,
                    # Rounded so a change of exactly the threshold is not lost to float error.
                    exceeds_threshold=bool(np.round(abs_change, 12) >= threshold),
                )
            )
    pairs.sort(key=lambda p: p.abs_change, reverse=True)
    return CorrelationReport(
        method=profile.method,
        threshold=float(threshold),
        columns=columns,
        pairs=pairs,
        undefined_pairs=undefined,
        top_k=top_k,
    )
