"""Performance-decay detection for fitted scikit-learn classifiers and regressors.

Drift metrics say the *inputs* changed; decay metrics say the *model got
worse*. Decay can only be measured once ground-truth labels for new data
arrive, which is often days or weeks later, so the two checks complement each
other.

Default decay thresholds for classifiers:

* accuracy: an absolute drop of 0.05 (5 percentage points) or more,
* F1: an absolute drop of 0.05 or more,
* log-loss: a relative increase of 20% or more (log-loss has no fixed scale,
  so a relative change is the meaningful comparison).

Default decay thresholds for regressors:

* R-squared: an absolute drop of 0.05 or more (R-squared is unit-free, so an
  absolute change means the same thing for every target),
* MAE and RMSE: a relative increase of 10% or more (both are in the units of
  the target, so only a relative change is comparable across models).

The regression defaults are starting points, not an industry convention; set
them to the error increase your use case can tolerate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import is_regressor
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score

CLASSIFICATION_METRICS = ("accuracy", "f1", "log_loss")
REGRESSION_METRICS = ("r2", "mae", "rmse")
METRICS = CLASSIFICATION_METRICS + REGRESSION_METRICS
#: For these metrics a larger value is better; for the rest, smaller is better.
HIGHER_IS_BETTER = {"accuracy": True, "f1": True, "log_loss": False, "r2": True, "mae": False, "rmse": False}
TASKS = ("classification", "regression")


@dataclass(frozen=True)
class DecayThresholds:
    max_accuracy_drop: float = 0.05
    max_f1_drop: float = 0.05
    max_log_loss_increase: float = 0.20
    max_r2_drop: float = 0.05
    max_mae_increase: float = 0.10
    max_rmse_increase: float = 0.10

    def __post_init__(self) -> None:
        for f in fields(self):
            if not getattr(self, f.name) >= 0:
                raise ValueError(f"{f.name} must be a non-negative number")

    def rule(self, metric: str) -> str:
        """Human-readable decay rule for ``metric``, e.g. ``'accuracy drop ≥ 0.05'``."""
        return {
            "accuracy": f"accuracy drop ≥ {self.max_accuracy_drop}",
            "f1": f"F1 drop ≥ {self.max_f1_drop}",
            "log_loss": f"log-loss increase ≥ {self.max_log_loss_increase:.0%}",
            "r2": f"R² drop ≥ {self.max_r2_drop}",
            "mae": f"MAE increase ≥ {self.max_mae_increase:.0%}",
            "rmse": f"RMSE increase ≥ {self.max_rmse_increase:.0%}",
        }[metric]

    def is_decayed(self, metric: str, change: float, relative: Optional[float]) -> bool:
        """Whether a ``change`` (current - reference) in ``metric`` counts as decay."""
        absolute_drop = {"accuracy": self.max_accuracy_drop, "f1": self.max_f1_drop, "r2": self.max_r2_drop}
        relative_increase = {
            "log_loss": self.max_log_loss_increase,
            "mae": self.max_mae_increase,
            "rmse": self.max_rmse_increase,
        }
        if metric in absolute_drop:
            return -change >= absolute_drop[metric]
        return relative is not None and relative >= relative_increase[metric]

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecayThresholds":
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in data.items() if k in known})


@dataclass
class MetricChange:
    metric: str
    reference: float
    current: float
    change: float
    relative_change: Optional[float]
    decayed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecayReport:
    reference: Dict[str, float]
    current: Dict[str, float]
    changes: List[MetricChange]
    decayed: bool
    thresholds: DecayThresholds = field(default_factory=DecayThresholds)

    @property
    def decayed_metrics(self) -> List[str]:
        return [c.metric for c in self.changes if c.decayed]

    @property
    def task(self) -> str:
        """``'regression'`` or ``'classification'``, judged from the metrics compared."""
        return "regression" if any(c.metric in REGRESSION_METRICS for c in self.changes) else "classification"

    @property
    def rules(self) -> List[str]:
        """The decay rules that applied to the compared metrics."""
        return [self.thresholds.rule(c.metric) for c in self.changes]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "current": self.current,
            "changes": [c.to_dict() for c in self.changes],
            "decayed": self.decayed,
            "decayed_metrics": self.decayed_metrics,
            "task": self.task,
            "thresholds": self.thresholds.to_dict(),
        }


def feature_columns(model: Any, df: pd.DataFrame, target: Optional[str] = None) -> List[str]:
    """Columns to feed the model: ``model.feature_names_in_`` if available, else every non-target column."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        cols = [str(c) for c in names]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"data is missing model feature columns: {missing}")
        return cols
    return [c for c in df.columns if c != target]


def infer_task(model: Any) -> str:
    """``'regression'`` for scikit-learn regressors (including pipelines ending in one), else ``'classification'``."""
    try:
        return "regression" if is_regressor(model) else "classification"
    except Exception:  # objects that only duck-type predict()
        return "classification"


def compute_metrics(
    model: Any,
    X: pd.DataFrame,
    y: Sequence[Any],
    task: Optional[str] = None,
) -> Dict[str, float]:
    """Score ``model`` on ``X`` / ``y``.

    For classifiers: accuracy, F1 and (if the model supports
    ``predict_proba``) log-loss. F1 is the positive-class F1 for binary
    problems (positive class = ``model.classes_[1]``) and macro-averaged F1
    otherwise.

    For regressors: R-squared, MAE and RMSE.

    ``task`` (``'classification'`` or ``'regression'``) is inferred from the
    model when not given; see :func:`infer_task`.
    """
    task = task or infer_task(model)
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    y_true = np.asarray(y)
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on an empty label set")
    if task == "regression":
        return regression_metrics(y_true, model.predict(X))
    y_pred = model.predict(X)
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = np.unique(np.concatenate([y_true, np.asarray(y_pred)]))
    classes = np.asarray(classes)

    if len(classes) == 2:
        f1 = f1_score(y_true, y_pred, pos_label=classes[1], average="binary", zero_division=0)
    else:
        f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1),
        "n": float(y_true.size),
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        result["log_loss"] = float(log_loss(y_true, proba, labels=classes))
    return result


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """R-squared, MAE and RMSE of ``y_pred`` against ``y_true``."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).reshape(y_true.shape)
    if y_true.size < 2:
        raise ValueError("regression metrics need at least 2 labelled rows")
    if np.isnan(y_pred).any():
        raise ValueError("model produced NaN predictions")
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n": float(y_true.size),
    }


def compare_metrics(
    reference: Dict[str, float],
    current: Dict[str, float],
    thresholds: Optional[DecayThresholds] = None,
) -> DecayReport:
    """Compare two metric dicts (as produced by :func:`compute_metrics`) and flag decay.

    Only metrics present in both dicts are compared, so classifier and
    regressor metric sets are handled by the same function.
    """
    thresholds = thresholds or DecayThresholds()
    changes: List[MetricChange] = []
    for metric in METRICS:
        if metric not in reference or metric not in current:
            continue
        ref, cur = float(reference[metric]), float(current[metric])
        change = cur - ref
        relative = change / abs(ref) if ref != 0 else None
        decayed = thresholds.is_decayed(metric, change, relative)
        changes.append(MetricChange(metric, ref, cur, change, relative, bool(decayed)))
    if not changes:
        raise ValueError(
            "no metrics in common between reference and current "
            f"(reference: {sorted(reference)}, current: {sorted(current)})"
        )
    return DecayReport(
        reference=dict(reference),
        current=dict(current),
        changes=changes,
        decayed=any(c.decayed for c in changes),
        thresholds=thresholds,
    )


def compare_performance(
    model: Any,
    X_reference: pd.DataFrame,
    y_reference: Sequence[Any],
    X_current: pd.DataFrame,
    y_current: Sequence[Any],
    thresholds: Optional[DecayThresholds] = None,
    task: Optional[str] = None,
) -> DecayReport:
    """Score ``model`` (classifier or regressor) on reference and current data and report any decay."""
    reference = compute_metrics(model, X_reference, y_reference, task)
    current = compute_metrics(model, X_current, y_current, task)
    return compare_metrics(reference, current, thresholds)
