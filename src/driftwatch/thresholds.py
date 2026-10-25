"""Severity thresholds and the rules that turn metrics into labels.

Defaults and their rationale (see README "Interpreting drift scores"):

* ``psi_moderate = 0.10`` and ``psi_high = 0.25`` are the long-standing credit
  risk conventions for the Population Stability Index (Siddiqi, *Credit Risk
  Scorecards*, 2006; also used by most model-risk teams): below 0.10 the
  population is considered stable, 0.10-0.25 warrants investigation, and 0.25
  or above is a significant shift that requires action.
* ``p_value = 0.01`` is the significance level for the KS (numeric) and
  chi-square (categorical) tests. A column whose PSI is below 0.10 but whose
  test still rejects "same distribution" is labelled ``low``: the change is
  statistically real but practically small. 0.01 rather than 0.05 keeps the
  false-alarm rate down when many columns are tested at once.
* ``drift_share = 0.30``: the dataset is flagged as drifted when at least 30%
  of monitored columns are ``moderate`` or worse, or any column is ``high``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

SEVERITIES = ("none", "low", "moderate", "high")
SEVERITY_RANK = {name: rank for rank, name in enumerate(SEVERITIES)}


@dataclass(frozen=True)
class DriftThresholds:
    psi_moderate: float = 0.10
    psi_high: float = 0.25
    p_value: float = 0.01
    drift_share: float = 0.30

    def __post_init__(self) -> None:
        if not 0 < self.psi_moderate < self.psi_high:
            raise ValueError("require 0 < psi_moderate < psi_high")
        if not 0 < self.p_value < 1:
            raise ValueError("p_value must be in (0, 1)")
        if not 0 < self.drift_share <= 1:
            raise ValueError("drift_share must be in (0, 1]")

    def severity(self, psi_value: float, p_value: Optional[float]) -> str:
        """Map a column's PSI and test p-value to ``none``/``low``/``moderate``/``high``."""
        if psi_value >= self.psi_high:
            return "high"
        if psi_value >= self.psi_moderate:
            return "moderate"
        if p_value is not None and p_value < self.p_value:
            return "low"
        return "none"

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriftThresholds":
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in data.items() if k in known})
