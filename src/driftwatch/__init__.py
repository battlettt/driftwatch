"""driftwatch: detect data drift and model decay in tabular ML models."""

__version__ = "0.1.0"

from .decay import DecayReport, DecayThresholds, compare_performance, compute_metrics  # noqa: E402
from .detector import ColumnDrift, DriftDetector, DriftReport  # noqa: E402
from .metrics import chi_square_test, js_divergence, ks_test, psi, total_variation  # noqa: E402
from .report import render_html, render_markdown  # noqa: E402
from .thresholds import DriftThresholds  # noqa: E402

__all__ = [
    "__version__",
    "ColumnDrift",
    "DecayReport",
    "DecayThresholds",
    "DriftDetector",
    "DriftReport",
    "DriftThresholds",
    "chi_square_test",
    "compare_performance",
    "compute_metrics",
    "js_divergence",
    "ks_test",
    "psi",
    "render_html",
    "render_markdown",
    "total_variation",
]
