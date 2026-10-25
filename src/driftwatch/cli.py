"""Command-line interface: ``python -m driftwatch {fit,check}``.

Exit codes for ``check``: 0 when the run succeeds (regardless of drift),
1 when ``--fail-on`` is set and the condition is met, 2 for usage or input errors.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from . import __version__
from .detector import DriftDetector
from .report import STATUS_TEXT, describe_change, render_html, render_markdown
from .thresholds import DriftThresholds


def _read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    return pd.read_csv(p)


def _load_model(path: str) -> Any:
    # Pickle can execute arbitrary code: only load model files you created or trust.
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="driftwatch",
        description="Detect data drift and model decay in tabular ML data.",
    )
    parser.add_argument("--version", action="version", version=f"driftwatch {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit a detector on reference (training) data and save it as JSON")
    fit.add_argument("--reference", required=True, help="CSV file with reference data")
    fit.add_argument("--out", required=True, help="where to write the detector JSON")
    fit.add_argument("--target", help="label column; excluded from drift checks and used for the model baseline")
    fit.add_argument("--model", help="pickled fitted sklearn classifier (trusted files only); requires --target")
    fit.add_argument("--bins", type=int, default=10, help="quantile bins for numeric columns (default: 10)")
    fit.add_argument("--max-categories", type=int, default=50, help="categories tracked per column (default: 50)")
    fit.add_argument("--sample-size", type=int, default=5000, help="reference values kept per column for KS (default: 5000)")
    fit.add_argument("--categorical", nargs="+", default=[], metavar="COL", help="treat these numeric columns as categorical")
    fit.add_argument("--exclude", nargs="+", default=[], metavar="COL", help="ignore these columns (IDs, timestamps, ...)")
    fit.add_argument("--psi-moderate", type=float, default=DriftThresholds.psi_moderate)
    fit.add_argument("--psi-high", type=float, default=DriftThresholds.psi_high)
    fit.add_argument("--p-value", type=float, default=DriftThresholds.p_value)
    fit.add_argument("--drift-share", type=float, default=DriftThresholds.drift_share)

    check = sub.add_parser("check", help="compare new data against a fitted detector")
    check.add_argument("--detector", required=True, help="detector JSON produced by 'fit'")
    check.add_argument("--data", required=True, help="CSV file with current data")
    check.add_argument("--report", help="write a report to this path")
    check.add_argument(
        "--format",
        choices=["html", "md", "json"],
        help="report format (default: inferred from --report extension, else html)",
    )
    check.add_argument("--model", help="pickled fitted classifier (trusted files only); adds a decay section")
    check.add_argument("--top", type=int, default=5, help="number of top drifted columns to list (default: 5)")
    check.add_argument(
        "--fail-on",
        choices=["drift", "warning", "decay"],
        help="exit with status 1 if the dataset drifted ('drift'), anything changed ('warning'), or the model decayed",
    )
    return parser


def _infer_format(report_path: Optional[str], explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if report_path:
        suffix = Path(report_path).suffix.lower()
        if suffix in (".md", ".markdown"):
            return "md"
        if suffix == ".json":
            return "json"
    return "html"


def _cmd_fit(args: argparse.Namespace) -> int:
    if args.model and not args.target:
        raise ValueError("--model requires --target")
    reference = _read_csv(args.reference)
    thresholds = DriftThresholds(
        psi_moderate=args.psi_moderate,
        psi_high=args.psi_high,
        p_value=args.p_value,
        drift_share=args.drift_share,
    )
    detector = DriftDetector(
        n_bins=args.bins,
        max_categories=args.max_categories,
        sample_size=args.sample_size,
        categorical_columns=args.categorical,
        exclude_columns=args.exclude,
        thresholds=thresholds,
    )
    model = _load_model(args.model) if args.model else None
    detector.fit(reference, target=args.target, model=model)
    detector.save(args.out)

    kinds = [p.kind for p in detector.profiles.values()]
    print(
        f"Fitted detector on {len(reference):,} rows: {kinds.count('numeric')} numeric and "
        f"{kinds.count('categorical')} categorical columns."
    )
    if detector.baseline:
        b = detector.baseline
        extra = f", log_loss={b['log_loss']:.4f}" if "log_loss" in b else ""
        print(f"Model baseline: accuracy={b['accuracy']:.4f}, f1={b['f1']:.4f}{extra}")
    print(f"Saved to {args.out}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    detector = DriftDetector.load(args.detector)
    current = _read_csv(args.data)
    model = _load_model(args.model) if args.model else None
    report = detector.drift_report(current, top_k=args.top, model=model)

    print(f"Status: {STATUS_TEXT[report.status]}")
    print(
        f"Overall drift score: {report.overall_score:.3f} | drifted columns: "
        f"{report.drift_share:.0%} of {len(report.columns)}"
    )
    if report.missing_columns:
        print(f"Missing columns: {', '.join(report.missing_columns)}")
    if report.top_drifted:
        print("Top drifted columns:")
        width = max(len(c.column) for c in report.top_drifted)
        for c in report.top_drifted:
            print(f"  {c.column:<{width}}  {c.severity:<8}  PSI={c.psi:.3f}  {describe_change(c)}")
    if report.performance is not None:
        perf = report.performance
        state = "DECAYED (" + ", ".join(perf.decayed_metrics) + ")" if perf.decayed else "within tolerance"
        print(f"Model performance: {state}")
        for ch in perf.changes:
            print(f"  {ch.metric:<9} {ch.reference:.4f} -> {ch.current:.4f} ({ch.change:+.4f})")

    if args.report:
        fmt = _infer_format(args.report, args.format)
        if fmt == "md":
            content = render_markdown(report)
        elif fmt == "json":
            content = report.to_json()
        else:
            content = render_html(report)
        Path(args.report).write_text(content, encoding="utf-8")
        print(f"Report written to {args.report}")

    if args.fail_on == "drift" and report.dataset_drift:
        return 1
    if args.fail_on == "warning" and report.status != "ok":
        return 1
    if args.fail_on == "decay" and report.performance is not None and report.performance.decayed:
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "check" and args.fail_on == "decay" and not args.model:
        parser.error("--fail-on decay requires --model")
    try:
        if args.command == "fit":
            return _cmd_fit(args)
        return _cmd_check(args)
    except (FileNotFoundError, ValueError, KeyError, pickle.UnpicklingError) as exc:
        print(f"driftwatch: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
