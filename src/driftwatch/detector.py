"""The :class:`DriftDetector`: fit on reference data, compare new data against it."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from . import __version__
from .config import ThresholdConfig, ThresholdSpec, resolve_thresholds
from .correlation import (
    METHODS as CORRELATION_METHODS,
    CorrelationProfile,
    CorrelationReport,
    compare_correlations,
    fit_correlation,
)
from .decay import DecayReport, DecayThresholds, compare_metrics, compute_metrics, feature_columns
from .metrics import (
    bin_counts,
    chi_square_test,
    js_divergence,
    ks_test,
    psi,
    quantile_edges,
    total_variation,
)
from .thresholds import SEVERITY_RANK, DriftThresholds

OTHER = "__other__"
MISSING = "__missing__"
FORMAT_VERSION = 1


# --------------------------------------------------------------------------- #
# Reference profiles
# --------------------------------------------------------------------------- #


@dataclass
class NumericProfile:
    """Reference distribution of a numeric column.

    ``proportions`` covers ``len(edges) + 1`` value bins followed by one
    missing-value bin. ``sample`` is a capped random subsample of the
    reference values used for the KS test.
    """

    edges: List[float]
    proportions: List[float]
    mean: float
    std: float
    min: float
    max: float
    missing_rate: float
    n: int
    sample: List[float]
    kind: str = "numeric"


@dataclass
class CategoricalProfile:
    """Reference distribution of a categorical column.

    ``proportions`` covers ``categories`` in order, then an ``__other__``
    bucket (rare or unseen values), then a ``__missing__`` bucket.
    """

    categories: List[str]
    proportions: List[float]
    missing_rate: float
    n: int
    kind: str = "categorical"

    @property
    def buckets(self) -> List[str]:
        return [*self.categories, OTHER, MISSING]


Profile = Union[NumericProfile, CategoricalProfile]


def _profile_from_dict(data: Dict[str, Any]) -> Profile:
    data = dict(data)
    kind = data.get("kind")
    if kind == "numeric":
        return NumericProfile(**data)
    if kind == "categorical":
        return CategoricalProfile(**data)
    raise ValueError(f"unknown column kind in detector file: {kind!r}")


def _as_category_strings(series: pd.Series) -> pd.Series:
    """Categorical values as strings, with missing values kept as NaN."""
    return series.astype(object).where(series.notna()).map(lambda v: v if pd.isna(v) else str(v))


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class ColumnDrift:
    column: str
    kind: str
    severity: str
    psi: float
    js_divergence: float
    n_reference: int
    n_current: int
    reference_missing_rate: float
    current_missing_rate: float
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    chi2_statistic: Optional[float] = None
    chi2_pvalue: Optional[float] = None
    total_variation: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def p_value(self) -> Optional[float]:
        return self.ks_pvalue if self.kind == "numeric" else self.chi2_pvalue

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftReport:
    columns: List[ColumnDrift]
    overall_score: float
    drift_share: float
    dataset_drift: bool
    status: str
    n_reference: int
    n_current: int
    thresholds: DriftThresholds
    top_k: int = 5
    missing_columns: List[str] = field(default_factory=list)
    performance: Optional[DecayReport] = None
    correlation: Optional[CorrelationReport] = None
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    @property
    def column_drift(self) -> bool:
        """True when the per-column rules alone flag the dataset (drift share or any ``high`` column)."""
        share = self.drift_share >= self.thresholds.drift_share
        return share or any(c.severity == "high" for c in self.columns)

    @property
    def correlation_drift(self) -> bool:
        """True when a pair of numeric columns' correlation moved by at least ``correlation_shift``."""
        return self.correlation is not None and self.correlation.drifted

    @property
    def top_drifted(self) -> List[ColumnDrift]:
        """Up to ``top_k`` columns with severity above ``none``, worst first."""
        drifted = [c for c in self.columns if c.severity != "none"]
        drifted.sort(key=lambda c: (SEVERITY_RANK[c.severity], c.psi), reverse=True)
        return drifted[: self.top_k]

    def column(self, name: str) -> ColumnDrift:
        for col in self.columns:
            if col.column == name:
                return col
        raise KeyError(name)

    def severity_counts(self) -> Dict[str, int]:
        counts = {s: 0 for s in SEVERITY_RANK}
        for col in self.columns:
            counts[col.severity] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "overall_score": self.overall_score,
            "drift_share": self.drift_share,
            "dataset_drift": self.dataset_drift,
            "column_drift": self.column_drift,
            "correlation_drift": self.correlation_drift,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "generated_at": self.generated_at,
            "thresholds": self.thresholds.to_dict(),
            "severity_counts": self.severity_counts(),
            "top_drifted": [c.column for c in self.top_drifted],
            "missing_columns": self.missing_columns,
            "columns": [c.to_dict() for c in self.columns],
            "performance": self.performance.to_dict() if self.performance else None,
            "correlation": self.correlation.to_dict() if self.correlation else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    def to_markdown(self) -> str:
        from .report import render_markdown

        return render_markdown(self)

    def to_html(self) -> str:
        from .report import render_html

        return render_html(self)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #


class DriftDetector:
    """Learns per-column reference distributions and scores new data against them.

    Parameters
    ----------
    n_bins:
        Number of quantile bins for numeric columns (fewer are used when the
        column has many tied values).
    max_categories:
        Most frequent categories tracked individually per categorical column;
        the rest share an ``__other__`` bucket.
    sample_size:
        Maximum number of reference values kept per numeric column for the KS
        test. Larger values give a more precise test and a larger JSON file.
    categorical_columns:
        Columns to force to categorical even if they are numeric (e.g. codes).
    exclude_columns:
        Columns to ignore entirely (IDs, timestamps, ...).
    thresholds:
        Severity thresholds (:class:`DriftThresholds`), or a full threshold
        config: a :class:`~driftwatch.config.ThresholdConfig`, a dict in
        config-file shape, or the path to a JSON config file (see
        :mod:`driftwatch.config`). A config's ``decay`` section sets the decay
        thresholds unless ``decay_thresholds`` is also given.
    decay_thresholds:
        Performance-decay thresholds; see :mod:`driftwatch.decay`.
    correlation_method:
        ``'pearson'`` (default) or ``'spearman'``, for the correlation-shift
        check across numeric columns; see :mod:`driftwatch.correlation`.
    random_state:
        Seed for the reference subsample, so fitting is reproducible.
    """

    def __init__(
        self,
        n_bins: int = 10,
        max_categories: int = 50,
        sample_size: int = 5000,
        categorical_columns: Optional[Iterable[str]] = None,
        exclude_columns: Optional[Iterable[str]] = None,
        thresholds: Optional[ThresholdSpec] = None,
        decay_thresholds: Optional[DecayThresholds] = None,
        random_state: int = 0,
        correlation_method: str = "pearson",
    ) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if max_categories < 1:
            raise ValueError("max_categories must be at least 1")
        if sample_size < 1:
            raise ValueError("sample_size must be at least 1")
        if correlation_method not in CORRELATION_METHODS:
            raise ValueError(f"correlation_method must be one of {CORRELATION_METHODS}")
        config = resolve_thresholds(thresholds)
        self.n_bins = int(n_bins)
        self.max_categories = int(max_categories)
        self.sample_size = int(sample_size)
        self.categorical_columns = list(categorical_columns or [])
        self.exclude_columns = list(exclude_columns or [])
        self.thresholds = config.drift
        self.decay_thresholds = decay_thresholds or config.decay
        self.random_state = int(random_state)
        self.correlation_method = correlation_method

        self.profiles: Dict[str, Profile] = {}
        self.correlation: Optional[CorrelationProfile] = None
        self.target: Optional[str] = None
        self.baseline: Optional[Dict[str, float]] = None
        self.feature_names: Optional[List[str]] = None
        self.n_reference: int = 0
        self.fitted_at: Optional[str] = None

    # ------------------------------------------------------------------ fit

    @property
    def is_fitted(self) -> bool:
        return bool(self.profiles)

    def _is_categorical(self, name: str, series: pd.Series) -> bool:
        if name in self.categorical_columns:
            return True
        return pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series)

    def fit(
        self,
        reference: pd.DataFrame,
        target: Optional[str] = None,
        model: Any = None,
    ) -> "DriftDetector":
        """Store reference distributions for every column of ``reference``.

        If ``target`` is given it is excluded from drift monitoring. If
        ``model`` (a fitted classifier or regressor) is given too, its metrics
        on the reference data are stored as the performance baseline:
        accuracy / F1 / log-loss for classifiers, R² / MAE / RMSE for
        regressors.

        The correlation matrix of the numeric columns is stored too, for the
        correlation-shift check.
        """
        if not isinstance(reference, pd.DataFrame):
            raise TypeError("reference must be a pandas DataFrame")
        if len(reference) == 0:
            raise ValueError("reference data is empty")
        if model is not None and target is None:
            raise ValueError("a target column is required to compute a model baseline")
        if target is not None and target not in reference.columns:
            raise ValueError(f"target column {target!r} not found in reference data")

        skip = set(self.exclude_columns) | ({target} if target else set())
        rng = np.random.default_rng(self.random_state)
        profiles: Dict[str, Profile] = {}
        for name in reference.columns:
            if name in skip:
                continue
            series = reference[name]
            key = str(name)
            if self._is_categorical(key, series):
                profiles[key] = self._fit_categorical(series)
            else:
                profiles[key] = self._fit_numeric(series, rng)
        if not profiles:
            raise ValueError("no columns left to monitor after excluding target/excluded columns")

        self.profiles = profiles
        numeric = [name for name, p in profiles.items() if isinstance(p, NumericProfile)]
        by_name = {str(c): c for c in reference.columns}
        self.correlation = fit_correlation(
            reference[[by_name[c] for c in numeric]].set_axis(numeric, axis=1), numeric, self.correlation_method
        )
        self.target = target
        self.n_reference = int(len(reference))
        self.fitted_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.baseline = None
        self.feature_names = None
        if model is not None:
            self.set_baseline(model, reference)
        return self

    def set_baseline(self, model: Any, labelled: pd.DataFrame) -> Dict[str, float]:
        """Compute and store the model's reference metrics on labelled data."""
        if self.target is None:
            raise ValueError("detector was fitted without a target column")
        if self.target not in labelled.columns:
            raise ValueError(f"target column {self.target!r} not found in data")
        cols = feature_columns(model, labelled, self.target)
        self.feature_names = cols
        self.baseline = compute_metrics(model, labelled[cols], labelled[self.target])
        return self.baseline

    def _fit_numeric(self, series: pd.Series, rng: np.random.Generator) -> NumericProfile:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        present = values[~np.isnan(values)]
        edges = quantile_edges(present, self.n_bins)
        counts = bin_counts(values, edges)
        if present.size > self.sample_size:
            sample = rng.choice(present, size=self.sample_size, replace=False)
        else:
            sample = present
        has = present.size > 0
        return NumericProfile(
            edges=[float(e) for e in edges],
            proportions=(counts / counts.sum()).tolist(),
            mean=float(present.mean()) if has else float("nan"),
            std=float(present.std(ddof=1)) if present.size > 1 else 0.0,
            min=float(present.min()) if has else float("nan"),
            max=float(present.max()) if has else float("nan"),
            missing_rate=float(np.isnan(values).mean()),
            n=int(values.size),
            sample=[float(v) for v in np.sort(sample)],
        )

    def _fit_categorical(self, series: pd.Series) -> CategoricalProfile:
        values = _as_category_strings(series)
        freq = values.value_counts(dropna=True)
        # Sort by count desc, then name, so ties are deterministic.
        ordered = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        categories = [k for k, _ in ordered[: self.max_categories]]
        profile = CategoricalProfile(
            categories=categories,
            proportions=[],
            missing_rate=float(values.isna().mean()),
            n=int(len(values)),
        )
        counts = self._categorical_counts(values, profile)
        profile.proportions = (counts / counts.sum()).tolist()
        return profile

    @staticmethod
    def _categorical_counts(values: pd.Series, profile: CategoricalProfile) -> np.ndarray:
        index = {c: i for i, c in enumerate(profile.categories)}
        counts = np.zeros(len(profile.categories) + 2, dtype=float)
        for value, count in values.value_counts(dropna=False).items():
            if pd.isna(value):
                counts[-1] += count
            else:
                counts[index.get(value, len(profile.categories))] += count
        return counts

    # ---------------------------------------------------------------- check

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("DriftDetector is not fitted; call fit() or load() first")

    def drift_report(
        self,
        current: pd.DataFrame,
        top_k: int = 5,
        model: Any = None,
        thresholds: Optional[ThresholdSpec] = None,
    ) -> DriftReport:
        """Compare ``current`` against the reference and return a :class:`DriftReport`.

        If ``model`` is given, the report also includes a performance-decay
        section; this requires a baseline stored at fit time and the target
        column to be present in ``current``.

        ``thresholds`` overrides the detector's stored thresholds for this
        report only. It accepts anything the constructor does, including the
        path to a JSON config file; dicts and files are partial overrides.
        """
        self._require_fitted()
        config = resolve_thresholds(thresholds, ThresholdConfig(self.thresholds, self.decay_thresholds))
        t = config.drift
        if not isinstance(current, pd.DataFrame):
            raise TypeError("current must be a pandas DataFrame")
        if len(current) == 0:
            raise ValueError("current data is empty")

        results: List[ColumnDrift] = []
        missing_columns: List[str] = []
        for name, profile in self.profiles.items():
            if name not in current.columns:
                missing_columns.append(name)
                continue
            if isinstance(profile, NumericProfile):
                results.append(self._check_numeric(name, profile, current[name], t))
            else:
                results.append(self._check_categorical(name, profile, current[name], t))

        if not results:
            raise ValueError("none of the reference columns are present in the current data")

        correlation = None
        if self.correlation is not None:
            correlation = compare_correlations(self.correlation, current, t.correlation_shift, top_k)

        drift_share = sum(SEVERITY_RANK[r.severity] >= SEVERITY_RANK["moderate"] for r in results) / len(results)
        any_high = any(r.severity == "high" for r in results)
        dataset_drift = drift_share >= t.drift_share or any_high or (correlation is not None and correlation.drifted)
        if dataset_drift:
            status = "drift"
        elif any(r.severity != "none" for r in results) or missing_columns:
            status = "warning"
        else:
            status = "ok"

        performance = None
        if model is not None:
            performance = self.performance_report(model, current, config.decay)

        return DriftReport(
            columns=results,
            overall_score=float(np.mean([min(r.psi / t.psi_high, 1.0) for r in results])),
            drift_share=float(drift_share),
            dataset_drift=bool(dataset_drift),
            status=status,
            n_reference=self.n_reference,
            n_current=int(len(current)),
            thresholds=t,
            top_k=top_k,
            missing_columns=missing_columns,
            performance=performance,
            correlation=correlation,
        )

    def performance_report(
        self,
        model: Any,
        current: pd.DataFrame,
        thresholds: Optional[ThresholdSpec] = None,
    ) -> DecayReport:
        """Score ``model`` on labelled ``current`` data and compare to the stored baseline.

        ``thresholds`` optionally overrides the stored decay thresholds (a
        :class:`DecayThresholds`, a config dict, or a config file path).
        """
        self._require_fitted()
        decay = resolve_thresholds(thresholds, ThresholdConfig(self.thresholds, self.decay_thresholds)).decay
        if self.baseline is None or self.target is None:
            raise ValueError("no performance baseline: fit() with target= and model=, or call set_baseline()")
        if self.target not in current.columns:
            raise ValueError(f"target column {self.target!r} not found in current data")
        cols = self.feature_names or feature_columns(model, current, self.target)
        labelled = current[current[self.target].notna()]
        metrics = compute_metrics(model, labelled[cols], labelled[self.target])
        return compare_metrics(self.baseline, metrics, decay)

    def _check_numeric(
        self, name: str, profile: NumericProfile, series: pd.Series, thresholds: Optional[DriftThresholds] = None
    ) -> ColumnDrift:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        present = values[~np.isnan(values)]
        counts = bin_counts(values, profile.edges)
        psi_value = psi(profile.proportions, counts)
        js_value = js_divergence(profile.proportions, counts)

        ks_stat = ks_p = None
        if present.size and profile.sample:
            ks_stat, ks_p = ks_test(profile.sample, present)

        details: Dict[str, Any] = {
            "reference_mean": profile.mean,
            "current_mean": float(present.mean()) if present.size else None,
            "reference_std": profile.std,
            "current_std": float(present.std(ddof=1)) if present.size > 1 else None,
            "reference_range": [profile.min, profile.max],
        }
        if details["current_mean"] is not None and profile.std > 0:
            details["mean_shift_std"] = (details["current_mean"] - profile.mean) / profile.std
        if present.size:
            details["share_outside_reference_range"] = float(
                np.mean((present < profile.min) | (present > profile.max))
            )

        return ColumnDrift(
            column=name,
            kind="numeric",
            severity=(thresholds or self.thresholds).severity(psi_value, ks_p),
            psi=psi_value,
            js_divergence=js_value,
            n_reference=profile.n,
            n_current=int(values.size),
            reference_missing_rate=profile.missing_rate,
            current_missing_rate=float(np.isnan(values).mean()),
            ks_statistic=ks_stat,
            ks_pvalue=ks_p,
            details=details,
        )

    def _check_categorical(
        self, name: str, profile: CategoricalProfile, series: pd.Series, thresholds: Optional[DriftThresholds] = None
    ) -> ColumnDrift:
        values = _as_category_strings(series)
        counts = self._categorical_counts(values, profile)
        psi_value = psi(profile.proportions, counts)
        js_value = js_divergence(profile.proportions, counts)
        tv_value = total_variation(profile.proportions, counts)
        reference_counts = np.asarray(profile.proportions) * profile.n
        chi2_stat, chi2_p = chi_square_test(reference_counts, counts)

        current_props = counts / counts.sum()
        shifts = [
            {"category": bucket, "reference_share": float(r), "current_share": float(c)}
            for bucket, r, c in zip(profile.buckets, profile.proportions, current_props)
            if r > 0 or c > 0
        ]
        shifts.sort(key=lambda s: abs(s["current_share"] - s["reference_share"]), reverse=True)

        known = set(profile.categories)
        unseen = values[values.notna() & ~values.isin(known)].value_counts()
        details = {
            "largest_shifts": shifts[:5],
            "unseen_categories": {str(k): int(v) for k, v in unseen.head(10).items()},
            "n_unseen_categories": int(unseen.size),
        }

        return ColumnDrift(
            column=name,
            kind="categorical",
            severity=(thresholds or self.thresholds).severity(psi_value, chi2_p),
            psi=psi_value,
            js_divergence=js_value,
            n_reference=profile.n,
            n_current=int(len(values)),
            reference_missing_rate=profile.missing_rate,
            current_missing_rate=float(values.isna().mean()),
            chi2_statistic=chi2_stat,
            chi2_pvalue=chi2_p,
            total_variation=tv_value,
            details=details,
        )

    # -------------------------------------------------------- serialisation

    def to_dict(self) -> Dict[str, Any]:
        self._require_fitted()
        return {
            "format": "driftwatch.detector",
            "format_version": FORMAT_VERSION,
            "driftwatch_version": __version__,
            "fitted_at": self.fitted_at,
            "config": {
                "n_bins": self.n_bins,
                "max_categories": self.max_categories,
                "sample_size": self.sample_size,
                "categorical_columns": self.categorical_columns,
                "exclude_columns": self.exclude_columns,
                "random_state": self.random_state,
                "correlation_method": self.correlation_method,
                "thresholds": self.thresholds.to_dict(),
                "decay_thresholds": self.decay_thresholds.to_dict(),
            },
            "n_reference": self.n_reference,
            "target": self.target,
            "feature_names": self.feature_names,
            "baseline": self.baseline,
            "columns": {name: asdict(p) for name, p in self.profiles.items()},
            "correlation": self.correlation.to_dict() if self.correlation else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriftDetector":
        if data.get("format") != "driftwatch.detector":
            raise ValueError("not a driftwatch detector file")
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported detector format version: {data.get('format_version')!r}")
        cfg = data["config"]
        det = cls(
            n_bins=cfg["n_bins"],
            max_categories=cfg["max_categories"],
            sample_size=cfg["sample_size"],
            categorical_columns=cfg.get("categorical_columns"),
            exclude_columns=cfg.get("exclude_columns"),
            thresholds=DriftThresholds.from_dict(cfg.get("thresholds", {})),
            decay_thresholds=DecayThresholds.from_dict(cfg.get("decay_thresholds", {})),
            random_state=cfg.get("random_state", 0),
            correlation_method=cfg.get("correlation_method", "pearson"),
        )
        det.profiles = {name: _profile_from_dict(p) for name, p in data["columns"].items()}
        # Detector files written before the correlation check existed have no "correlation" key.
        corr = data.get("correlation")
        det.correlation = CorrelationProfile.from_dict(corr) if corr else None
        det.n_reference = int(data["n_reference"])
        det.target = data.get("target")
        det.feature_names = data.get("feature_names")
        det.baseline = data.get("baseline")
        det.fitted_at = data.get("fitted_at")
        return det

    def to_json(self, indent: Optional[int] = None) -> str:
        # NaN (e.g. the mean of an all-missing column) is written as null so the
        # file stays strict JSON.
        return json.dumps(_nan_to_none(self.to_dict()), indent=indent, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "DriftDetector":
        return cls.from_dict(_none_to_nan(json.loads(text)))

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DriftDetector":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


_NAN_FIELDS = ("mean", "min", "max")


def _nan_to_none(obj: Any) -> Any:
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _none_to_nan(data: Dict[str, Any]) -> Dict[str, Any]:
    for profile in data.get("columns", {}).values():
        for key in _NAN_FIELDS:
            if key in profile and profile[key] is None:
                profile[key] = float("nan")
    return data
