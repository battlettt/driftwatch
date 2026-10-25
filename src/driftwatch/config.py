"""Load drift and decay thresholds from a JSON config file.

A config file has up to two sections; every key is optional and anything left
out keeps its current value (the detector's stored thresholds, or the
defaults)::

    {
      "drift": {"psi_moderate": 0.1, "psi_high": 0.25, "p_value": 0.01,
                "drift_share": 0.3, "correlation_shift": 0.2},
      "decay": {"max_accuracy_drop": 0.05, "max_f1_drop": 0.05,
                "max_log_loss_increase": 0.2, "max_r2_drop": 0.05,
                "max_mae_increase": 0.1, "max_rmse_increase": 0.1}
    }

Unknown sections or keys are rejected, so a typo cannot silently fall back
to a default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union

from .decay import DecayThresholds
from .thresholds import DriftThresholds

T = TypeVar("T", DriftThresholds, DecayThresholds)

#: Anything accepted where thresholds can be given: a full config, one of the
#: two threshold objects, a dict in config-file shape, or a path to a JSON file.
ThresholdSpec = Union["ThresholdConfig", DriftThresholds, DecayThresholds, Dict[str, Any], str, Path]


@dataclass(frozen=True)
class ThresholdConfig:
    drift: DriftThresholds = field(default_factory=DriftThresholds)
    decay: DecayThresholds = field(default_factory=DecayThresholds)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {"drift": self.drift.to_dict(), "decay": self.decay.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base: Optional["ThresholdConfig"] = None) -> "ThresholdConfig":
        """Apply the values in ``data`` on top of ``base`` (defaults if omitted)."""
        base = base or cls()
        if not isinstance(data, dict):
            raise ValueError("threshold config must be a JSON object")
        unknown = sorted(set(data) - {"drift", "decay"})
        if unknown:
            raise ValueError(f"unknown threshold config section(s) {unknown}; expected 'drift' and/or 'decay'")
        return cls(
            drift=_override(base.drift, data.get("drift", {}), "drift"),
            decay=_override(base.decay, data.get("decay", {}), "decay"),
        )

    @classmethod
    def load(cls, path: Union[str, Path], base: Optional["ThresholdConfig"] = None) -> "ThresholdConfig":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"config file {path} is not valid JSON: {exc}") from exc
        return cls.from_dict(data, base)


def _override(current: T, values: Any, section: str) -> T:
    if not isinstance(values, dict):
        raise ValueError(f"config section {section!r} must be a JSON object")
    known = [f.name for f in fields(current)]
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"unknown {section} threshold(s) {unknown}; allowed: {known}")
    parsed: Dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{section}.{key} must be a number, got {value!r}")
        parsed[key] = float(value)
    # replace() re-runs the dataclass validation, so out-of-range values are rejected here.
    return replace(current, **parsed)


def resolve_thresholds(spec: Optional[ThresholdSpec], base: Optional[ThresholdConfig] = None) -> ThresholdConfig:
    """Turn any accepted threshold ``spec`` into a :class:`ThresholdConfig`.

    Dicts and config files are partial overrides of ``base``; a
    :class:`DriftThresholds` or :class:`DecayThresholds` replaces only its own
    half; a :class:`ThresholdConfig` replaces both.
    """
    base = base or ThresholdConfig()
    if spec is None:
        return base
    if isinstance(spec, ThresholdConfig):
        return spec
    if isinstance(spec, DriftThresholds):
        return replace(base, drift=spec)
    if isinstance(spec, DecayThresholds):
        return replace(base, decay=spec)
    if isinstance(spec, dict):
        return ThresholdConfig.from_dict(spec, base)
    if isinstance(spec, (str, Path)):
        return ThresholdConfig.load(spec, base)
    raise TypeError(f"unsupported thresholds specification: {type(spec).__name__}")
