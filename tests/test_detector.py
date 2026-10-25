import json

import numpy as np
import pandas as pd
import pytest

from driftwatch import DriftDetector, DriftThresholds
from driftwatch.detector import MISSING, OTHER


def test_fit_profiles_numeric_and_categorical(reference_df):
    det = DriftDetector().fit(reference_df, target="y")
    assert set(det.profiles) == {"x1", "x2", "color"}
    assert det.profiles["x1"].kind == "numeric"
    assert det.profiles["color"].kind == "categorical"
    assert det.profiles["color"].categories == ["red", "green", "blue"]
    for profile in det.profiles.values():
        assert sum(profile.proportions) == pytest.approx(1.0)
    x1 = det.profiles["x1"]
    assert len(x1.edges) == 9
    assert x1.mean == pytest.approx(reference_df["x1"].mean())
    assert x1.std == pytest.approx(reference_df["x1"].std())


def test_same_distribution_reports_no_drift(reference_df, stable_df):
    report = DriftDetector().fit(reference_df, target="y").drift_report(stable_df)
    assert report.status == "ok"
    assert not report.dataset_drift
    assert report.top_drifted == []
    assert all(c.psi < 0.1 for c in report.columns)


def test_identical_data_scores_zero(reference_df):
    report = DriftDetector().fit(reference_df).drift_report(reference_df)
    for col in report.columns:
        assert col.psi == pytest.approx(0.0, abs=1e-12)
        assert col.severity == "none"
    assert report.overall_score == pytest.approx(0.0, abs=1e-12)


def test_shifted_data_is_flagged(reference_df, drifted_df):
    report = DriftDetector().fit(reference_df, target="y").drift_report(drifted_df)
    assert report.dataset_drift
    assert report.status == "drift"
    x1 = report.column("x1")
    assert x1.severity == "high"
    assert x1.ks_pvalue < 1e-6
    assert x1.details["mean_shift_std"] == pytest.approx(1.5, abs=0.15)
    # x2 was not shifted
    assert report.column("x2").severity in ("none", "low")
    assert report.top_drifted[0].column in ("x1", "color")
    assert "x2" not in [c.column for c in report.top_drifted if c.severity == "high"]


def test_categorical_drift(reference_df, drifted_df):
    report = DriftDetector().fit(reference_df, target="y").drift_report(drifted_df)
    color = report.column("color")
    assert color.kind == "categorical"
    assert color.severity == "high"
    assert color.chi2_pvalue < 1e-10
    assert color.total_variation == pytest.approx(0.4, abs=0.05)
    assert color.ks_statistic is None
    top = color.details["largest_shifts"][0]
    assert top["category"] in ("red", "blue")


def test_unseen_category_goes_to_other_bucket():
    ref = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 50})
    cur = pd.DataFrame({"c": ["a"] * 40 + ["b"] * 40 + ["z"] * 20})
    report = DriftDetector().fit(ref).drift_report(cur)
    col = report.column("c")
    assert col.details["unseen_categories"] == {"z": 20}
    shares = {s["category"]: s["current_share"] for s in col.details["largest_shifts"]}
    assert shares[OTHER] == pytest.approx(0.2)
    assert col.severity == "high"


def test_max_categories_rolls_rare_values_into_other():
    ref = pd.DataFrame({"c": list("aaaabbbccd")})
    det = DriftDetector(max_categories=2).fit(ref)
    profile = det.profiles["c"]
    assert profile.categories == ["a", "b"]
    assert profile.proportions[2] == pytest.approx(0.3)  # c, c, d -> __other__


def test_missing_values_are_tracked():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"v": rng.normal(size=2000), "c": rng.choice(["a", "b"], 2000)})
    cur = ref.copy()
    cur.loc[:599, "v"] = np.nan
    cur.loc[:599, "c"] = None
    report = DriftDetector().fit(ref).drift_report(cur)
    v, c = report.column("v"), report.column("c")
    assert v.current_missing_rate == pytest.approx(0.3)
    assert v.severity == "high"
    assert c.severity == "high"
    assert any(s["category"] == MISSING for s in c.details["largest_shifts"])


def test_all_missing_current_column_does_not_crash():
    ref = pd.DataFrame({"v": np.arange(100, dtype=float)})
    cur = pd.DataFrame({"v": [np.nan] * 50})
    col = DriftDetector().fit(ref).drift_report(cur).column("v")
    assert col.ks_statistic is None
    assert col.severity == "high"


def test_missing_column_is_reported(reference_df, stable_df):
    report = DriftDetector().fit(reference_df, target="y").drift_report(stable_df.drop(columns=["x2"]))
    assert report.missing_columns == ["x2"]
    assert report.status == "warning"


def test_options_exclude_and_force_categorical():
    df = pd.DataFrame({"id": range(100), "code": [1, 2, 3, 4] * 25, "v": np.linspace(0, 1, 100)})
    det = DriftDetector(exclude_columns=["id"], categorical_columns=["code"]).fit(df)
    assert set(det.profiles) == {"code", "v"}
    assert det.profiles["code"].kind == "categorical"
    assert det.profiles["code"].categories == ["1", "2", "3", "4"]


def test_custom_thresholds_change_severity(reference_df, stable_df):
    default = DriftDetector().fit(reference_df, target="y").drift_report(stable_df)
    assert default.status == "ok"
    lowest = min(c.psi for c in default.columns)
    strict = DriftThresholds(psi_moderate=lowest / 4, psi_high=lowest / 2)
    report = DriftDetector(thresholds=strict).fit(reference_df, target="y").drift_report(stable_df)
    assert all(c.severity == "high" for c in report.columns)
    assert report.status == "drift"


def test_threshold_severity_rules():
    t = DriftThresholds()
    assert t.severity(0.30, 0.5) == "high"
    assert t.severity(0.12, 0.5) == "moderate"
    assert t.severity(0.05, 0.001) == "low"
    assert t.severity(0.05, 0.5) == "none"
    assert t.severity(0.05, None) == "none"
    with pytest.raises(ValueError):
        DriftThresholds(psi_moderate=0.3, psi_high=0.2)


def test_errors_on_bad_usage(reference_df):
    with pytest.raises(RuntimeError):
        DriftDetector().drift_report(reference_df)
    with pytest.raises(ValueError):
        DriftDetector().fit(reference_df.iloc[:0])
    with pytest.raises(ValueError):
        DriftDetector().fit(reference_df, target="nope")
    det = DriftDetector().fit(reference_df)
    with pytest.raises(ValueError):
        det.drift_report(pd.DataFrame({"other": [1, 2]}))


# --------------------------------------------------------- serialisation


def test_serialization_round_trip(tmp_path, reference_df, drifted_df, model):
    det = DriftDetector(n_bins=8, categorical_columns=[], thresholds=DriftThresholds(p_value=0.05))
    det.fit(reference_df, target="y", model=model)
    path = tmp_path / "detector.json"
    det.save(path)

    raw = json.loads(path.read_text())
    assert raw["format"] == "driftwatch.detector"
    loaded = DriftDetector.load(path)

    assert loaded.n_bins == 8
    assert loaded.thresholds == det.thresholds
    assert loaded.target == "y"
    assert loaded.baseline == det.baseline
    assert loaded.feature_names == ["x1", "x2"]
    assert loaded.profiles == det.profiles

    a = det.drift_report(drifted_df).to_dict()
    b = loaded.drift_report(drifted_df).to_dict()
    for key in ("generated_at",):
        a.pop(key), b.pop(key)
    assert a == b


def test_serialization_handles_all_missing_reference_column(tmp_path):
    df = pd.DataFrame({"v": [np.nan] * 10, "w": np.arange(10.0)})
    det = DriftDetector().fit(df)
    text = det.to_json()
    assert "NaN" not in text  # strict JSON
    loaded = DriftDetector.from_json(text)
    assert np.isnan(loaded.profiles["v"].mean)


def test_from_dict_rejects_foreign_files():
    with pytest.raises(ValueError):
        DriftDetector.from_dict({"format": "something-else"})


def test_report_to_json_is_valid(reference_df, drifted_df):
    report = DriftDetector().fit(reference_df, target="y").drift_report(drifted_df)
    data = json.loads(report.to_json())
    assert data["status"] == "drift"
    assert data["top_drifted"][0] in ("x1", "color")
    assert len(data["columns"]) == 3
