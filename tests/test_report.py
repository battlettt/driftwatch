import re

import numpy as np

import pandas as pd

from driftwatch import DriftDetector, render_html, render_markdown


def test_markdown_report_contents(reference_df, drifted_df):
    report = DriftDetector().fit(reference_df, target="y").drift_report(drifted_df)
    md = render_markdown(report)
    assert md.startswith("# driftwatch report")
    assert "DRIFT DETECTED" in md
    assert "## Top drifted columns" in md
    assert "## All columns" in md
    assert "`x1`" in md and "`color`" in md
    assert "## Model performance" not in md


def test_markdown_escapes_pipes_in_column_names():
    ref = pd.DataFrame({"a|b": ["x", "y"] * 50})
    cur = pd.DataFrame({"a|b": ["x"] * 90 + ["y"] * 10})
    md = render_markdown(DriftDetector().fit(ref).drift_report(cur))
    assert "`a\\|b`" in md


def test_html_report_is_self_contained(reference_df, drifted_df, model):
    det = DriftDetector().fit(reference_df, target="y", model=model)
    html = render_html(det.drift_report(drifted_df, model=model))
    assert html.startswith("<!DOCTYPE html>")
    assert "<style>" in html
    assert "Model performance" in html
    # no external resources of any kind
    assert not re.search(r"https?://", html)
    assert "<link" not in html and "<script" not in html and "@import" not in html and "src=" not in html


def test_html_escapes_untrusted_values():
    ref = pd.DataFrame({"<script>": ["<b>x</b>"] * 50 + ["y"] * 50})
    cur = pd.DataFrame({"<script>": ["<img src=x>"] * 100})
    html = render_html(DriftDetector().fit(ref).drift_report(cur))
    assert "<script>" not in html.split("</style>", 1)[1]
    assert "<img" not in html
    assert "&lt;script&gt;" in html


def test_clean_report_says_ok(reference_df, stable_df):
    report = DriftDetector().fit(reference_df, target="y").drift_report(stable_df)
    assert "No column drifted" in render_markdown(report)
    assert 'class="banner ok"' in render_html(report)


def _scrambled_pair_frames():
    rng = np.random.default_rng(0)
    a = rng.normal(size=3000)
    ref = pd.DataFrame({"a": a, "b<i>": a + rng.normal(scale=0.5, size=3000)})
    cur = ref.sample(2000, random_state=1).reset_index(drop=True)
    cur["b<i>"] = rng.permutation(cur["b<i>"].to_numpy())
    return ref, cur


def test_markdown_correlation_section():
    ref, cur = _scrambled_pair_frames()
    report = DriftDetector().fit(ref).drift_report(cur)
    md = render_markdown(report)
    assert "## Correlation shifts" in md
    assert "**1 column pair changed correlation by 0.2 or more" in md
    assert re.search(r"\| `a` × `b<i>` \| \+0\.8\d\d \| \+0\.0\d\d \| -0\.8\d\d \| yes \|", md)
    assert "| yes |" in md
    assert "Largest correlation shift" in md
    assert "correlation of any pair of numeric columns changes by 0.2" in md


def test_html_correlation_section_is_escaped_and_self_contained():
    ref, cur = _scrambled_pair_frames()
    html = render_html(DriftDetector().fit(ref).drift_report(cur))
    assert "<h2>Correlation shifts</h2>" in html
    assert "Max correlation shift" in html
    assert "<code>b&lt;i&gt;</code>" in html
    assert "<i>" not in html.split("</style>", 1)[1].replace("<i style", "")
    assert not re.search(r"https?://", html)


def test_report_without_numeric_pairs_has_no_correlation_section():
    ref = pd.DataFrame({"c": ["x", "y"] * 50, "v": range(100)})
    report = DriftDetector().fit(ref).drift_report(ref)
    assert "Correlation shifts" not in render_markdown(report)
    assert "Correlation shifts" not in render_html(report)


def test_clean_report_shows_correlation_ok(reference_df, stable_df):
    report = DriftDetector().fit(reference_df, target="y").drift_report(stable_df)
    md = render_markdown(report)
    assert "No column pair changed correlation by 0.2 or more" in md
    assert "| `x1` × `x2` |" in md
