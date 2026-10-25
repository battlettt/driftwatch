"""Render a :class:`~driftwatch.detector.DriftReport` as Markdown or self-contained HTML.

The HTML output uses inline CSS only and loads nothing from the network, so it
can be attached to a ticket, emailed, or opened from an air-gapped machine.
"""

from __future__ import annotations

import math
from html import escape
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .detector import ColumnDrift, DriftReport

STATUS_TEXT = {
    "ok": "OK: no meaningful drift",
    "warning": "WARNING: some columns changed",
    "drift": "DRIFT DETECTED",
}


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def _num(value: Optional[float], digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def _pvalue(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "-"
    if value < 1e-4:
        return "<0.0001"
    return f"{value:.4f}"


def _pct(value: Optional[float], digits: int = 1) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{100 * value:.{digits}f}%"


def _compact(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def describe_change(col: "ColumnDrift") -> str:
    """One-line, human-readable summary of what changed in a column."""
    d = col.details
    parts: List[str] = []
    if col.kind == "numeric":
        if d.get("current_mean") is not None:
            text = f"mean {_compact(d['reference_mean'])} → {_compact(d['current_mean'])}"
            if d.get("mean_shift_std") is not None:
                text += f" ({d['mean_shift_std']:+.2f} σ)"
            parts.append(text)
        outside = d.get("share_outside_reference_range")
        if outside:
            parts.append(f"{_pct(outside)} outside reference range")
    else:
        shifts = d.get("largest_shifts") or []
        if shifts:
            s = shifts[0]
            parts.append(f"'{s['category']}' {_pct(s['reference_share'])} → {_pct(s['current_share'])}")
        n_unseen = d.get("n_unseen_categories", 0)
        if n_unseen:
            names = ", ".join(f"'{k}'" for k in list(d["unseen_categories"])[:3])
            parts.append(f"{n_unseen} unseen categor{'y' if n_unseen == 1 else 'ies'} ({names})")
    if abs(col.current_missing_rate - col.reference_missing_rate) >= 0.01:
        parts.append(f"missing {_pct(col.reference_missing_rate)} → {_pct(col.current_missing_rate)}")
    return "; ".join(parts) if parts else "-"


def _md_cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _summary_rows(report: "DriftReport") -> List[tuple]:
    counts = report.severity_counts()
    t = report.thresholds
    return [
        ("Status", STATUS_TEXT[report.status]),
        ("Overall drift score", f"{report.overall_score:.3f} (0 = identical, 1 = every column at/above PSI {t.psi_high})"),
        (
            "Drifted columns (moderate or high)",
            f"{counts['moderate'] + counts['high']} of {len(report.columns)} ({_pct(report.drift_share, 0)})",
        ),
        ("Severity counts", ", ".join(f"{k}: {v}" for k, v in counts.items())),
        ("Reference rows", f"{report.n_reference:,}"),
        ("Current rows", f"{report.n_current:,}"),
        ("Generated", report.generated_at),
    ]


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def render_markdown(report: "DriftReport") -> str:
    t = report.thresholds
    lines: List[str] = ["# driftwatch report", ""]
    lines += ["| | |", "|---|---|"]
    lines += [f"| **{k}** | {_md_cell(v)} |" for k, v in _summary_rows(report)]
    lines.append("")

    if report.missing_columns:
        lines += [
            "> **Missing columns:** "
            + ", ".join(f"`{c}`" for c in report.missing_columns)
            + " were present in the reference data but not in the current data.",
            "",
        ]

    lines += ["## Top drifted columns", ""]
    top = report.top_drifted
    if top:
        lines += ["| Column | Type | Severity | PSI | What changed |", "|---|---|---|---:|---|"]
        for c in top:
            lines.append(
                f"| `{_md_cell(c.column)}` | {c.kind} | **{c.severity}** | {_num(c.psi)} | {_md_cell(describe_change(c))} |"
            )
    else:
        lines.append("No column drifted beyond the `none` severity level.")
    lines.append("")

    lines += [
        "## All columns",
        "",
        "| Column | Type | Severity | PSI | JS div. | KS stat | KS p | χ² p | TV dist. | Missing (ref → cur) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in sorted(report.columns, key=lambda c: c.psi, reverse=True):
        lines.append(
            f"| `{_md_cell(c.column)}` | {c.kind} | {c.severity} | {_num(c.psi)} | {_num(c.js_divergence)} "
            f"| {_num(c.ks_statistic)} | {_pvalue(c.ks_pvalue)} | {_pvalue(c.chi2_pvalue)} "
            f"| {_num(c.total_variation)} | {_pct(c.reference_missing_rate)} → {_pct(c.current_missing_rate)} |"
        )
    lines.append("")

    if report.performance is not None:
        perf = report.performance
        verdict = (
            "**Model performance has decayed** (" + ", ".join(perf.decayed_metrics) + ")."
            if perf.decayed
            else "Model performance is within tolerance."
        )
        lines += ["## Model performance", "", verdict, ""]
        lines += ["| Metric | Reference | Current | Change | Decayed |", "|---|---:|---:|---:|---|"]
        for ch in perf.changes:
            rel = f" ({ch.relative_change:+.1%})" if ch.relative_change is not None else ""
            lines.append(
                f"| {ch.metric} | {ch.reference:.4f} | {ch.current:.4f} | {ch.change:+.4f}{rel} | {'yes' if ch.decayed else 'no'} |"
            )
        dt = perf.thresholds
        lines += [
            "",
            f"Decay rules: accuracy drop ≥ {dt.max_accuracy_drop}, F1 drop ≥ {dt.max_f1_drop}, "
            f"log-loss increase ≥ {dt.max_log_loss_increase:.0%}. "
            f"Scored on {int(perf.current.get('n', 0)):,} labelled rows.",
            "",
        ]

    lines += [
        "## How to read this report",
        "",
        f"- **Severity** comes from the Population Stability Index: `high` if PSI ≥ {t.psi_high}, "
        f"`moderate` if PSI ≥ {t.psi_moderate}, `low` if PSI is below {t.psi_moderate} but the KS "
        f"(numeric) or chi-square (categorical) test rejects equality at p < {t.p_value}, otherwise `none`.",
        f"- The dataset is flagged as drifted when at least {t.drift_share:.0%} of columns are `moderate` "
        "or worse, or any column is `high`.",
        "- **JS div.** is the Jensen-Shannon divergence (base 2, 0 to 1) over the same bins as PSI. "
        "**TV dist.** is the share of probability mass that moved between categories.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

_CSS = """
:root { --fg:#1f2328; --muted:#59636e; --border:#d1d9e0; --bg:#ffffff; --panel:#f6f8fa;
        --none:#1a7f37; --low:#0969da; --moderate:#9a6700; --high:#cf222e; }
* { box-sizing: border-box; }
body { margin:0; padding:32px; background:var(--bg); color:var(--fg);
       font: 14px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 32px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
.sub { color: var(--muted); margin: 0 0 20px; }
.banner { padding: 14px 18px; border-radius: 8px; font-weight: 600; font-size: 16px; margin-bottom: 20px;
          border: 1px solid; }
.banner.ok { color: var(--none); border-color: var(--none); background: #dafbe1; }
.banner.warning { color: var(--moderate); border-color: var(--moderate); background: #fff8c5; }
.banner.drift { color: var(--high); border-color: var(--high); background: #ffebe9; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.tile { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
.tile .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.tile .value { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
th { background: var(--panel); font-weight: 600; white-space: nowrap; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px; font-weight: 600;
         color: #fff; }
.badge.none { background: var(--none); } .badge.low { background: var(--low); }
.badge.moderate { background: var(--moderate); } .badge.high { background: var(--high); }
.bar { position: relative; height: 8px; width: 120px; background: #eaeef2; border-radius: 4px; margin-top: 4px; }
.bar span { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; }
.bar i { position: absolute; top: -2px; bottom: -2px; width: 1px; background: var(--muted); }
.notice { background: #fff8c5; border: 1px solid var(--moderate); border-radius: 6px; padding: 10px 14px; }
ul.notes { color: var(--muted); padding-left: 20px; }
footer { color: var(--muted); font-size: 12px; margin-top: 32px; }
"""

_SEVERITY_COLOR = {"none": "var(--none)", "low": "var(--low)", "moderate": "var(--moderate)", "high": "var(--high)"}


def _badge(severity: str) -> str:
    return f'<span class="badge {escape(severity)}">{escape(severity)}</span>'


def _flag(decayed: bool) -> str:
    return '<span class="badge high">yes</span>' if decayed else '<span class="badge none">no</span>'


def _psi_bar(col: "ColumnDrift", psi_moderate: float, psi_high: float) -> str:
    # The bar spans 0 .. 2 * psi_high; tick marks show the moderate/high thresholds.
    scale = 2 * psi_high
    width = min(col.psi / scale, 1.0) * 100
    ticks = "".join(f'<i style="left:{100 * th / scale:.1f}%"></i>' for th in (psi_moderate, psi_high))
    return (
        f'<div class="bar" title="PSI {col.psi:.3f}"><span style="width:{width:.1f}%;'
        f'background:{_SEVERITY_COLOR[col.severity]}"></span>{ticks}</div>'
    )


def render_html(report: "DriftReport") -> str:
    t = report.thresholds
    counts = report.severity_counts()
    out: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>driftwatch report</title>",
        f"<style>{_CSS}</style></head><body><main>",
        "<h1>driftwatch report</h1>",
        f'<p class="sub">Generated {escape(report.generated_at)} · '
        f"{report.n_reference:,} reference rows vs {report.n_current:,} current rows</p>",
        f'<div class="banner {escape(report.status)}">{escape(STATUS_TEXT[report.status])}</div>',
        '<div class="tiles">',
    ]
    tiles = [
        ("Overall drift score", f"{report.overall_score:.3f}"),
        ("Drifted columns", f"{_pct(report.drift_share, 0)}"),
        ("Columns monitored", str(len(report.columns))),
        ("High / moderate", f"{counts['high']} / {counts['moderate']}"),
    ]
    if report.performance is not None:
        tiles.append(("Model decay", "yes" if report.performance.decayed else "no"))
    for label, value in tiles:
        out.append(f'<div class="tile"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>')
    out.append("</div>")

    if report.missing_columns:
        cols = ", ".join(f"<code>{escape(c)}</code>" for c in report.missing_columns)
        out.append(
            f'<h2>Missing columns</h2><p class="notice">{cols} were present in the reference data '
            "but not in the current data.</p>"
        )

    out.append("<h2>Top drifted columns</h2>")
    top = report.top_drifted
    if top:
        out.append("<table><thead><tr><th>Column</th><th>Type</th><th>Severity</th><th>PSI</th><th>What changed</th></tr></thead><tbody>")
        for c in top:
            out.append(
                f"<tr><td><code>{escape(c.column)}</code></td><td>{c.kind}</td><td>{_badge(c.severity)}</td>"
                f'<td class="num">{_num(c.psi)}{_psi_bar(c, t.psi_moderate, t.psi_high)}</td>'
                f"<td>{escape(describe_change(c))}</td></tr>"
            )
        out.append("</tbody></table>")
    else:
        out.append("<p>No column drifted beyond the <code>none</code> severity level.</p>")

    out.append("<h2>All columns</h2>")
    out.append(
        "<table><thead><tr><th>Column</th><th>Type</th><th>Severity</th><th>PSI</th><th>JS div.</th>"
        "<th>KS stat</th><th>KS p</th><th>χ² p</th><th>TV dist.</th><th>Missing (ref → cur)</th></tr></thead><tbody>"
    )
    for c in sorted(report.columns, key=lambda c: c.psi, reverse=True):
        out.append(
            f"<tr><td><code>{escape(c.column)}</code></td><td>{c.kind}</td><td>{_badge(c.severity)}</td>"
            f'<td class="num">{_num(c.psi)}{_psi_bar(c, t.psi_moderate, t.psi_high)}</td>'
            f'<td class="num">{_num(c.js_divergence)}</td><td class="num">{_num(c.ks_statistic)}</td>'
            f'<td class="num">{_pvalue(c.ks_pvalue)}</td><td class="num">{_pvalue(c.chi2_pvalue)}</td>'
            f'<td class="num">{_num(c.total_variation)}</td>'
            f'<td class="num">{_pct(c.reference_missing_rate)} → {_pct(c.current_missing_rate)}</td></tr>'
        )
    out.append("</tbody></table>")

    if report.performance is not None:
        perf = report.performance
        out.append("<h2>Model performance</h2>")
        if perf.decayed:
            out.append(
                f'<div class="banner drift">Model performance has decayed ({escape(", ".join(perf.decayed_metrics))})</div>'
            )
        else:
            out.append('<div class="banner ok">Model performance is within tolerance</div>')
        out.append(
            "<table><thead><tr><th>Metric</th><th>Reference</th><th>Current</th><th>Change</th>"
            "<th>Decayed</th></tr></thead><tbody>"
        )
        for ch in perf.changes:
            rel = f" ({ch.relative_change:+.1%})" if ch.relative_change is not None else ""
            out.append(
                f"<tr><td>{escape(ch.metric)}</td><td class=\"num\">{ch.reference:.4f}</td>"
                f'<td class="num">{ch.current:.4f}</td><td class="num">{ch.change:+.4f}{rel}</td>'
                f"<td>{_flag(ch.decayed)}</td></tr>"
            )
        out.append("</tbody></table>")
        dt = perf.thresholds
        out.append(
            f'<p class="sub">Decay rules: accuracy drop ≥ {dt.max_accuracy_drop}, F1 drop ≥ {dt.max_f1_drop}, '
            f"log-loss increase ≥ {dt.max_log_loss_increase:.0%}. "
            f"Scored on {int(perf.current.get('n', 0)):,} labelled rows.</p>"
        )

    out += [
        "<h2>How to read this report</h2>",
        '<ul class="notes">',
        f"<li><b>Severity</b> comes from the Population Stability Index: <code>high</code> if PSI ≥ {t.psi_high}, "
        f"<code>moderate</code> if PSI ≥ {t.psi_moderate}, <code>low</code> if PSI is below {t.psi_moderate} "
        f"but the KS (numeric) or chi-square (categorical) test rejects equality at p &lt; {t.p_value}, "
        "otherwise <code>none</code>. Tick marks on the PSI bars show the two thresholds.</li>",
        f"<li>The dataset is flagged as drifted when at least {t.drift_share:.0%} of columns are "
        "<code>moderate</code> or worse, or any column is <code>high</code>.</li>",
        "<li><b>JS div.</b> is the Jensen-Shannon divergence (base 2, 0 to 1) over the same bins as PSI. "
        "<b>TV dist.</b> is the share of probability mass that moved between categories.</li>",
        "</ul>",
        "<footer>Generated by driftwatch. This file is self-contained and makes no network requests.</footer>",
        "</main></body></html>",
    ]
    return "\n".join(out)
