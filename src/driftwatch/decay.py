"""Performance-decay detection for fitted scikit-learn classifiers.

Drift metrics say the *inputs* changed; decay metrics say the *model got
worse*. Decay can only be measured once ground-truth labels for new data
arrive, which is often days or weeks later, so the two checks complement each
other.

Default decay thresholds:

* accuracy: an absolute drop of 0.05 (5 percentage points) or more,
* F1: an absolute drop of 0.05 or more,
* log-loss: a relative increase of 20% or more (log-loss has no fixed scale,
  so a relative change is the meaningful comparison).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

METRICS = ("accuracy", "f1", "log_loss")
#: For these metrics a larger value is better; for the rest, smaller is better.
HIGHER_IS_BETTER = {"accuracy": True, "f1": True, "log_loss": False}


@dataclass(frozen=True)
class DecayThresholds:
    max_accuracy_drop: float = 0.05
    max_f1_drop: float = 0.05
    max_log_loss_increase: float = 0.20

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "current": self.current,
            "changes": [c.to_dict() for c in self.changes],
            "decayed": self.decayed,
            "decayed_metrics": self.decayed_metrics,
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


def compute_metrics(model: Any, X: pd.DataFrame, y: Sequence[Any]) -> Dict[str, float]:
    """Accuracy, F1 and (if the model supports ``predict_proba``) log-loss.

    F1 is the positive-class F1 for binary problems (positive class =
    ``model.classes_[1]``) and macro-averaged F1 otherwise.
    """
    y_true = np.asarray(y)
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on an empty label set")
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


def compare_metrics(
    reference: Dict[str, float],
    current: Dict[str, float],
    thresholds: Optional[DecayThresholds] = None,
) -> DecayReport:
    """Compare two metric dicts (as produced by :func:`compute_metrics`) and flag decay."""
    thresholds = thresholds or DecayThresholds()
    changes: List[MetricChange] = []
    for metric in METRICS:
        if metric not in reference or metric not in current:
            continue
        ref, cur = float(reference[metric]), float(current[metric])
        change = cur - ref
        relative = change / ref if ref != 0 else None
        if metric == "accuracy":
            decayed = -change >= thresholds.max_accuracy_drop
        elif metric == "f1":
            decayed = -change >= thresholds.max_f1_drop
        else:
            decayed = relative is not None and relative >= thresholds.max_log_loss_increase
        changes.append(MetricChange(metric, ref, cur, change, relative, bool(decayed)))
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
) -> DecayReport:
    """Score ``model`` on reference and current data and report any decay."""
    reference = compute_metrics(model, X_reference, y_reference)
    current = compute_metrics(model, X_current, y_current)
    return compare_metrics(reference, current, thresholds)
