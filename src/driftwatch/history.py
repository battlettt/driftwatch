"""Check history: append each check result to a JSONL file, list it, and chart it.

Each line of a history file is one JSON object describing one check (see
:class:`HistoryEntry`). Lines are appended in the order checks run, so the
file is chronological; :func:`read_history` returns entries in file order
(oldest first). Appending opens the file in append mode and writes one line,
so the file can be committed, tailed or concatenated with ordinary tools.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from . import __version__
from .report import _CSS, STATUS_TEXT

if TYPE_CHECKING:  # pragma: no cover
    from .detector import DriftReport

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_STATUSES = tuple(STATUS_TEXT)


@dataclass
class HistoryEntry:
    """One recorded check. ``top_drifted`` items are ``{"column", "severity", "psi"}`` dicts."""

    timestamp: str
    status: str
    overall_score: float
    drift_share: float
    dataset_drift: bool
    correlation_drift: bool
    top_drifted: List[Dict[str, Any]] = field(default_factory=list)
    max_correlation_shift: Optional[float] = None
    model_decayed: Optional[bool] = None
    n_current: Optional[int] = None
    missing_columns: List[str] = field(default_factory=list)
    source: Optional[str] = None
    driftwatch_version: str = __version__

    @property
    def time(self) -> datetime:
        return datetime.strptime(self.timestamp, _TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"))

    @classmethod
    def from_report(
        cls,
        report: "DriftReport",
        source: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> "HistoryEntry":
        when = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
        corr = report.correlation
        max_shift = corr.max_abs_change if corr is not None and corr.pairs else None
        return cls(
            timestamp=when.strftime(_TIMESTAMP_FORMAT),
            status=report.status,
            overall_score=round(float(report.overall_score), 6),
            drift_share=round(float(report.drift_share), 6),
            dataset_drift=bool(report.dataset_drift),
            correlation_drift=bool(report.correlation_drift),
            top_drifted=[
                {"column": c.column, "severity": c.severity, "psi": round(float(c.psi), 6)}
                for c in report.top_drifted
            ],
            max_correlation_shift=None if max_shift is None else round(float(max_shift), 6),
            model_decayed=None if report.performance is None else bool(report.performance.decayed),
            n_current=int(report.n_current),
            missing_columns=list(report.missing_columns),
            source=source,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryEntry":
        if not isinstance(data, dict):
            raise ValueError("history entry must be a JSON object")
        missing = [k for k in ("timestamp", "status", "overall_score") if k not in data]
        if missing:
            raise ValueError(f"history entry is missing {', '.join(missing)}")
        if data["status"] not in _STATUSES:
            raise ValueError(f"unknown status {data['status']!r}")
        score = data["overall_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"overall_score must be a number, got {score!r}")
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        known.setdefault("drift_share", 0.0)
        known.setdefault("dataset_drift", data["status"] == "drift")
        known.setdefault("correlation_drift", False)
        entry = cls(**known)
        entry.time  # validates the timestamp format
        return entry


def append_history(
    path: Union[str, Path],
    report: "DriftReport",
    source: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> HistoryEntry:
    """Append ``report`` to the JSONL history file at ``path`` (created if needed) and return the entry."""
    entry = HistoryEntry.from_report(report, source=source, timestamp=timestamp)
    p = Path(path)
    if p.parent and not p.parent.exists():
        raise FileNotFoundError(f"directory not found: {p.parent}")
    with p.open("a", encoding="utf-8") as fh:
        fh.write(entry.to_json() + "\n")
    return entry


def read_history(path: Union[str, Path]) -> List[HistoryEntry]:
    """Read every entry from a history file, oldest first (file order). Blank lines are skipped."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    entries: List[HistoryEntry] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(HistoryEntry.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}, line {lineno}: invalid history entry ({exc})") from None
    return entries


# --------------------------------------------------------------------------- #
# Text table
# --------------------------------------------------------------------------- #


def _yes_no(value: Optional[bool]) -> str:
    return "-" if value is None else ("yes" if value else "no")


def _top_names(entry: HistoryEntry, limit: int = 3) -> str:
    names = [str(t.get("column")) for t in entry.top_drifted[:limit]]
    extra = len(entry.top_drifted) - limit
    if extra > 0:
        names.append(f"+{extra} more")
    return ", ".join(names) if names else "-"


def _rows(entries: Sequence[HistoryEntry], numbers: Sequence[int]) -> List[List[str]]:
    return [
        [
            str(n),
            e.time.strftime("%Y-%m-%d %H:%M:%S"),
            e.status,
            f"{e.overall_score:.3f}",
            f"{e.drift_share:.0%}",
            _yes_no(e.correlation_drift),
            _yes_no(e.model_decayed),
            _top_names(e),
        ]
        for n, e in zip(numbers, entries)
    ]


_HEADERS = ["#", "Time (UTC)", "Status", "Score", "Drifted", "Corr. drift", "Decayed", "Top drifted columns"]
_NUMERIC_COLS = {0, 3, 4}


def format_history_table(entries: Sequence[HistoryEntry], newest_first: bool = False, first_number: int = 1) -> str:
    """Plain-text table of ``entries``, numbered by check order starting at ``first_number``.

    Pass ``first_number`` when ``entries`` is a tail of the history so rows keep their original numbers.
    """
    if not entries:
        return "No checks recorded."
    numbers = list(range(first_number, first_number + len(entries)))
    rows = _rows(entries, numbers)
    if newest_first:
        rows.reverse()
    widths = [max(len(r[i]) for r in [_HEADERS, *rows]) for i in range(len(_HEADERS))]

    def fmt(row: List[str]) -> str:
        cells = [c.rjust(w) if i in _NUMERIC_COLS else c.ljust(w) for i, (c, w) in enumerate(zip(row, widths))]
        return "  ".join(cells).rstrip()

    return "\n".join([fmt(_HEADERS), *(fmt(r) for r in rows)])


# --------------------------------------------------------------------------- #
# HTML trend page
# --------------------------------------------------------------------------- #

_TREND_CSS = """
.chart { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
.chart svg { display: block; width: 100%; height: auto; }
.chart .grid { stroke: var(--border); stroke-width: 1; }
.chart .axis { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.chart .line { fill: none; stroke: var(--fg); stroke-width: 2; stroke-linejoin: round; }
.chart circle { stroke: var(--panel); stroke-width: 2; }
.chart circle.ok { fill: var(--none); } .chart circle.warning { fill: var(--moderate); }
.chart circle.drift { fill: var(--high); }
.chart g.pt:hover circle { stroke: var(--fg); }
.chart .hit { fill: transparent; stroke: none; }
.legend { display: flex; gap: 16px; color: var(--muted); font-size: 12px; margin-top: 8px; flex-wrap: wrap; }
.legend .key::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%;
                       margin-right: 6px; vertical-align: -1px; background: var(--dot); }
.badge.ok { background: var(--none); } .badge.warning { background: var(--moderate); }
.badge.drift { background: var(--high); }
"""

_W, _H = 900, 280
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 44, 16, 14, 34


def _x_positions(entries: Sequence[HistoryEntry]) -> List[float]:
    """X pixel per entry: proportional to time, or evenly spaced if all checks share a timestamp."""
    left, right = _PAD_L + 8, _W - _PAD_R - 8
    if len(entries) == 1:
        return [(left + right) / 2]
    times = [e.time.timestamp() for e in entries]
    t0, t1 = min(times), max(times)
    if t1 == t0:
        return [left + (right - left) * i / (len(entries) - 1) for i in range(len(entries))]
    return [left + (right - left) * (t - t0) / (t1 - t0) for t in times]


def _y(score: float) -> float:
    clamped = min(max(score, 0.0), 1.0)
    return _PAD_T + (_H - _PAD_T - _PAD_B) * (1 - clamped)


def _chart_svg(entries: Sequence[HistoryEntry], first_number: int = 1) -> str:
    xs = _x_positions(entries)
    ys = [_y(e.overall_score) for e in entries]
    out = [
        f'<svg viewBox="0 0 {_W} {_H}" role="img" aria-labelledby="chart-title">',
        '<title id="chart-title">Overall drift score per check (0 = no drift, 1 = every column at the high PSI '
        "threshold)</title>",
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = _y(tick)
        out.append(f'<line class="grid" x1="{_PAD_L}" x2="{_W - _PAD_R}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="axis" x="{_PAD_L - 6}" y="{y + 4:.1f}" text-anchor="end">{tick:.2f}</text>')

    # Step line: a score holds from one check until the next one replaces it.
    points = [f"{xs[0]:.1f},{ys[0]:.1f}"]
    for i in range(1, len(entries)):
        points.append(f"{xs[i]:.1f},{ys[i - 1]:.1f}")
        points.append(f"{xs[i]:.1f},{ys[i]:.1f}")
    if len(entries) > 1:
        out.append(f'<polyline class="line" points="{" ".join(points)}"/>')

    for i, (e, x, y) in enumerate(zip(entries, xs, ys), start=first_number):
        tip = (
            f"#{i} {e.time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n{STATUS_TEXT[e.status]}\n"
            f"score {e.overall_score:.3f}, drifted columns {e.drift_share:.0%}"
        )
        if e.correlation_drift:
            tip += "\ncorrelation drift"
        if e.top_drifted:
            tip += "\ntop: " + _top_names(e)
        out.append(
            f'<g class="pt"><title>{escape(tip)}</title>'
            f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="12"/>'
            f'<circle class="{escape(e.status)}" cx="{x:.1f}" cy="{y:.1f}" r="5"/></g>'
        )

    base = _H - _PAD_B
    first, last = entries[0], entries[-1]
    # Short spans (e.g. several checks in one session) need seconds to tell the end labels apart.
    label = "%Y-%m-%d %H:%M:%S" if (last.time - first.time).total_seconds() < 3600 else "%Y-%m-%d %H:%M"
    out.append(f'<text class="axis" x="{xs[0]:.1f}" y="{base + 20}" text-anchor="start">'
               f"{escape(first.time.strftime(label))}</text>")
    if len(entries) > 1:
        out.append(f'<text class="axis" x="{xs[-1]:.1f}" y="{base + 20}" text-anchor="end">'
                   f"{escape(last.time.strftime(label))}</text>")
    out.append("</svg>")
    return "".join(out)


def render_trend_html(
    entries: Sequence[HistoryEntry], title: str = "driftwatch trend", first_number: int = 1
) -> str:
    """Self-contained HTML page: overall drift score over time plus a table of the checks.

    Checks are numbered from ``first_number`` (see :func:`format_history_table`).

    Uses inline CSS and inline SVG only; nothing is loaded from the network.
    """
    if not entries:
        raise ValueError("history has no checks to plot")
    counts = {s: sum(e.status == s for e in entries) for s in _STATUSES}
    latest = entries[-1]
    out: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(title)}</title>",
        f"<style>{_CSS}{_TREND_CSS}</style></head><body><main>",
        f"<h1>{escape(title)}</h1>",
        f'<p class="sub">{len(entries)} check{"" if len(entries) == 1 else "s"} from '
        f"{escape(entries[0].time.strftime('%Y-%m-%d %H:%M:%S'))} to "
        f"{escape(latest.time.strftime('%Y-%m-%d %H:%M:%S'))} UTC</p>",
        f'<div class="banner {escape(latest.status)}">Latest check: {escape(STATUS_TEXT[latest.status])}</div>',
        '<div class="tiles">',
    ]
    tiles = [
        ("Latest score", f"{latest.overall_score:.3f}"),
        ("Highest score", f"{max(e.overall_score for e in entries):.3f}"),
        ("Checks", str(len(entries))),
        ("OK / warning / drift", f"{counts['ok']} / {counts['warning']} / {counts['drift']}"),
    ]
    for label, value in tiles:
        out.append(f'<div class="tile"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>')
    out.append("</div>")

    out.append("<h2>Overall drift score over time</h2>")
    out.append('<div class="chart">' + _chart_svg(entries, first_number))
    out.append(
        '<div class="legend"><span class="key" style="--dot:var(--none)">ok</span>'
        '<span class="key" style="--dot:var(--moderate)">warning</span><span class="key" style="--dot:var(--high)">drift</span>'
        "<span>Hover a point for details. Points are placed by check time; the line holds each score "
        "until the next check.</span></div></div>"
    )

    out.append("<h2>Checks</h2>")
    out.append(
        "<table><thead><tr>" + "".join(f"<th>{escape(h)}</th>" for h in [*_HEADERS[:7], "Max corr. shift", _HEADERS[7], "Data"])
        + "</tr></thead><tbody>"
    )
    for i, e in enumerate(entries, start=first_number):
        shift = "-" if e.max_correlation_shift is None else f"{e.max_correlation_shift:.3f}"
        top = ", ".join(
            f"<code>{escape(str(t.get('column')))}</code> ({escape(str(t.get('severity')))})" for t in e.top_drifted
        ) or "-"
        out.append(
            f'<tr><td class="num">{i}</td><td>{escape(e.time.strftime("%Y-%m-%d %H:%M:%S"))}</td>'
            f'<td><span class="badge {escape(e.status)}">{escape(e.status)}</span></td>'
            f'<td class="num">{e.overall_score:.3f}</td><td class="num">{e.drift_share:.0%}</td>'
            f"<td>{_yes_no(e.correlation_drift)}</td><td>{_yes_no(e.model_decayed)}</td>"
            f'<td class="num">{shift}</td><td>{top}</td><td>{escape(e.source or "-")}</td></tr>'
        )
    out.append("</tbody></table>")
    out += [
        '<ul class="notes"><li><b>Score</b> is the overall drift score: the mean over columns of '
        "<code>min(PSI / psi_high, 1)</code>. 0 means nothing moved; 1 means every column is at or above the "
        "<code>high</code> PSI threshold.</li>"
        "<li><b>Status</b> also reflects the drift-share and correlation-shift rules, so a check can be "
        "<code>drift</code> with a low score.</li></ul>",
        "<footer>Generated by driftwatch. This file is self-contained and makes no network requests.</footer>",
        "</main></body></html>",
    ]
    return "\n".join(out)
