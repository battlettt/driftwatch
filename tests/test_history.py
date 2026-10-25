import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import driftwatch
from driftwatch import DriftDetector, HistoryEntry, append_history, read_history, render_trend_html
from driftwatch.cli import main
from driftwatch.history import format_history_table

T0 = datetime(2026, 1, 5, 9, 30, tzinfo=timezone.utc)


@pytest.fixture
def reports(reference_df, stable_df, drifted_df, model):
    det = DriftDetector().fit(reference_df, target="y", model=model)
    return det.drift_report(stable_df, model=model), det.drift_report(drifted_df, model=model)


@pytest.fixture
def csvs(tmp_path, reference_df, stable_df, drifted_df):
    paths = {}
    for name, df in {"ref": reference_df, "stable": stable_df, "drift": drifted_df}.items():
        paths[name] = tmp_path / f"{name}.csv"
        df.to_csv(paths[name], index=False)
    return paths


# --------------------------------------------------------------------------- library


def test_append_writes_one_json_line_per_check(tmp_path, reports):
    stable, drifted = reports
    path = tmp_path / "history.jsonl"
    append_history(path, stable, source="stable.csv", timestamp=T0)
    entry = append_history(path, drifted, source="drift.csv", timestamp=T0 + timedelta(days=1))

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["timestamp"] == "2026-01-05T09:30:00Z"
    assert first["status"] == "ok" and first["correlation_drift"] is False
    assert second["status"] == "drift"
    assert second["overall_score"] == pytest.approx(drifted.overall_score, abs=1e-6)
    assert [t["column"] for t in second["top_drifted"]] == [c.column for c in drifted.top_drifted]
    assert second["top_drifted"][0]["column"] == "x1" and second["top_drifted"][0]["severity"] == "high"
    assert second["model_decayed"] is drifted.performance.decayed
    assert second["source"] == "drift.csv"
    assert second["driftwatch_version"] == driftwatch.__version__
    assert entry.to_dict() == second


def test_read_history_round_trip_in_file_order(tmp_path, reports):
    stable, drifted = reports
    path = tmp_path / "h.jsonl"
    for i, rep in enumerate([stable, drifted, stable]):
        append_history(path, rep, timestamp=T0 + timedelta(hours=i))
    entries = read_history(path)
    assert [e.status for e in entries] == ["ok", "drift", "ok"]
    assert entries[1].time == T0 + timedelta(hours=1)
    assert entries[1].correlation_drift == drifted.correlation_drift


def test_correlation_drift_flag_recorded(tmp_path):
    rng = pd.Series(range(400), dtype=float)
    ref = pd.DataFrame({"a": rng, "b": rng * 2 + (rng % 7)})
    cur = ref.copy()
    cur["b"] = cur["b"].sample(frac=1, random_state=0).to_numpy()  # same marginal, relationship broken
    report = DriftDetector().fit(ref).drift_report(cur)
    entry = append_history(tmp_path / "h.jsonl", report)
    assert entry.correlation_drift is True and entry.status == "drift"
    assert entry.max_correlation_shift > 0.9


def test_read_history_skips_blank_lines_and_rejects_bad_ones(tmp_path, reports):
    path = tmp_path / "h.jsonl"
    append_history(path, reports[0], timestamp=T0)
    path.write_text(path.read_text() + "\n\n")
    assert len(read_history(path)) == 1

    path.write_text(path.read_text() + "{not json\n")
    with pytest.raises(ValueError, match="line 4"):
        read_history(path)

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"timestamp": "2026-01-01T00:00:00Z", "status": "weird", "overall_score": 0.1}) + "\n")
    with pytest.raises(ValueError, match="unknown status"):
        read_history(bad)
    bad.write_text(json.dumps({"timestamp": "yesterday", "status": "ok", "overall_score": 0.1}) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        read_history(bad)
    bad.write_text(json.dumps({"status": "ok"}) + "\n")
    with pytest.raises(ValueError, match="missing timestamp, overall_score"):
        read_history(bad)

    with pytest.raises(FileNotFoundError):
        read_history(tmp_path / "nope.jsonl")


def test_entry_tolerates_unknown_keys_and_minimal_records():
    entry = HistoryEntry.from_dict(
        {"timestamp": "2026-01-01T00:00:00Z", "status": "drift", "overall_score": 0.5, "future_field": 1}
    )
    assert entry.dataset_drift is True and entry.top_drifted == [] and entry.model_decayed is None


def test_format_history_table_orders_and_numbers(tmp_path, reports):
    stable, drifted = reports
    entries = [
        HistoryEntry.from_report(stable, timestamp=T0),
        HistoryEntry.from_report(drifted, timestamp=T0 + timedelta(days=1)),
    ]
    table = format_history_table(entries).splitlines()
    assert table[0].split()[:4] == ["#", "Time", "(UTC)", "Status"]
    assert table[1].lstrip().startswith("1  2026-01-05 09:30:00  ok")
    assert table[2].lstrip().startswith("2  2026-01-06 09:30:00  drift")
    assert "x1" in table[2]

    reversed_table = format_history_table(entries, newest_first=True).splitlines()
    assert reversed_table[1].lstrip().startswith("2  2026-01-06")  # numbering stays chronological
    assert format_history_table([]) == "No checks recorded."


def test_trend_html_is_self_contained_and_escaped(reports):
    stable, drifted = reports
    entries = [HistoryEntry.from_report(stable, timestamp=T0, source="<b>evil</b>.csv")]
    entries += [HistoryEntry.from_report(drifted, timestamp=T0 + timedelta(days=d)) for d in (1, 3)]
    html = render_trend_html(entries)
    assert html.startswith("<!DOCTYPE html>")
    assert "<svg" in html and "<polyline" in html
    assert html.count('<circle class="drift"') == 2 and html.count('<circle class="ok"') == 1
    assert "&lt;b&gt;evil&lt;/b&gt;.csv" in html and "<b>evil</b>" not in html
    assert not re.search(r"https?://(?!www\.w3\.org)", html)
    assert "<link" not in html and "<script" not in html and "@import" not in html and "src=" not in html


def test_trend_x_positions_follow_time(reports):
    stable = reports[0]
    entries = [HistoryEntry.from_report(stable, timestamp=T0 + timedelta(days=d)) for d in (0, 1, 10)]
    xs = [float(x) for x in re.findall(r'<circle class="ok" cx="([\d.]+)"', render_trend_html(entries))]
    assert xs[0] < xs[1] < xs[2]
    assert (xs[1] - xs[0]) < (xs[2] - xs[1]) / 5  # 1 day vs 9 days apart

    same_time = [HistoryEntry.from_report(stable, timestamp=T0) for _ in range(3)]
    xs = [float(x) for x in re.findall(r'<circle class="ok" cx="([\d.]+)"', render_trend_html(same_time))]
    assert xs[1] - xs[0] == pytest.approx(xs[2] - xs[1])  # evenly spaced fallback


def test_trend_single_entry_and_empty(reports):
    html = render_trend_html([HistoryEntry.from_report(reports[0], timestamp=T0)])
    assert "<polyline" not in html and '<circle class="ok"' in html
    with pytest.raises(ValueError):
        render_trend_html([])


# --------------------------------------------------------------------------- CLI


def _fit(tmp_path, csvs):
    det = tmp_path / "detector.json"
    assert main(["fit", "--reference", str(csvs["ref"]), "--out", str(det), "--target", "y"]) == 0
    return det


def test_cli_check_history_then_history_and_trend(tmp_path, csvs, capsys):
    det = _fit(tmp_path, csvs)
    hist = tmp_path / "history.jsonl"
    base = ["check", "--detector", str(det), "--history", str(hist), "--data"]
    assert main(base + [str(csvs["stable"])]) == 0
    # history is recorded even when --fail-on trips
    assert main(base + [str(csvs["drift"]), "--fail-on", "drift"]) == 1
    assert f"Check appended to {hist}" in capsys.readouterr().out
    assert [e.status for e in read_history(hist)] == ["ok", "drift"]
    assert read_history(hist)[1].source == str(csvs["drift"])

    assert main(["history", "--history", str(hist)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and " ok " in lines[1] and " drift " in lines[2]

    assert main(["history", "--history", str(hist), "--newest-first"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert " drift " in lines[1] and " ok " in lines[2]

    assert main(["history", "--history", str(hist), "--last", "1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2 and " drift " in lines[1]
    assert lines[1].startswith("2  ")  # keeps its position in the full history

    assert main(["trend", "--history", str(hist), "--report", str(tmp_path / "last.html"), "--last", "1"]) == 0
    assert "#2 " in (tmp_path / "last.html").read_text()

    page = tmp_path / "trend.html"
    assert main(["trend", "--history", str(hist), "--report", str(page), "--title", "loans"]) == 0
    assert "Trend of 2 checks written to" in capsys.readouterr().out
    html = page.read_text()
    assert "<title>loans</title>" in html and html.count("<circle class=\"hit\"") == 2


def test_cli_history_errors(tmp_path, csvs, capsys):
    missing = tmp_path / "none.jsonl"
    assert main(["history", "--history", str(missing)]) == 2
    assert "file not found" in capsys.readouterr().err
    assert main(["trend", "--history", str(missing), "--report", str(tmp_path / "t.html")]) == 2

    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert main(["history", "--history", str(empty)]) == 0
    assert "No checks recorded." in capsys.readouterr().out
    assert main(["trend", "--history", str(empty), "--report", str(tmp_path / "t.html")]) == 2
    assert "no checks" in capsys.readouterr().err
    assert main(["history", "--history", str(empty), "--last", "0"]) == 2

    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text("[1, 2]\n")
    assert main(["history", "--history", str(corrupt)]) == 2
    assert "line 1" in capsys.readouterr().err

    det = _fit(tmp_path, csvs)
    bad_dir = tmp_path / "no_such_dir" / "h.jsonl"
    assert main(["check", "--detector", str(det), "--data", str(csvs["stable"]), "--history", str(bad_dir)]) == 2


def test_history_commands_via_python_dash_m(tmp_path, csvs):
    det = _fit(tmp_path, csvs)
    hist = tmp_path / "h.jsonl"
    run = lambda *a: subprocess.run([sys.executable, "-m", "driftwatch", *a], capture_output=True, text=True)  # noqa: E731
    assert run("check", "--detector", str(det), "--data", str(csvs["stable"]), "--history", str(hist)).returncode == 0
    out = run("history", "--history", str(hist))
    assert out.returncode == 0, out.stderr
    assert "Time (UTC)" in out.stdout


def test_version_is_consistent():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1) == driftwatch.__version__ == "0.2.0"
    changelog = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text()
    assert f"## [{driftwatch.__version__}]" in changelog
