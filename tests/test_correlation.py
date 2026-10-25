import json

import numpy as np
import pandas as pd
import pytest

from driftwatch import DriftDetector, DriftThresholds, compare_correlations
from driftwatch.correlation import MIN_PAIR_ROWS, CorrelationProfile, fit_correlation


def correlated_frame(n: int, seed: int, rho: float = 0.8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rho * a + np.sqrt(1 - rho**2) * rng.normal(size=n)
    c = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "cat": rng.choice(["x", "y"], n)})


def scrambled(df: pd.DataFrame, column: str, seed: int = 0) -> pd.DataFrame:
    """Same marginal distributions, but ``column`` is shuffled across rows."""
    out = df.copy()
    out[column] = np.random.default_rng(seed).permutation(out[column].to_numpy())
    return out


def test_fit_stores_numeric_correlation_matrix():
    ref = correlated_frame(3000, seed=1)
    det = DriftDetector().fit(ref)
    corr = det.correlation
    assert corr.columns == ["a", "b", "c"]  # categorical column excluded
    assert corr.method == "pearson"
    m = np.array(corr.matrix)
    assert m.shape == (3, 3)
    assert np.allclose(np.diag(m), 1.0)
    assert m[0, 1] == pytest.approx(ref["a"].corr(ref["b"]))


def test_stable_batch_has_no_correlation_drift():
    det = DriftDetector().fit(correlated_frame(3000, seed=1))
    report = det.drift_report(correlated_frame(2000, seed=2))
    assert report.correlation is not None
    assert not report.correlation.drifted
    assert not report.correlation_drift
    assert report.correlation.max_abs_change < 0.1
    assert report.status == "ok"
    assert len(report.correlation.pairs) == 3


def test_broken_relationship_is_flagged_even_though_columns_look_unchanged():
    det = DriftDetector().fit(correlated_frame(3000, seed=1))
    report = det.drift_report(scrambled(correlated_frame(2000, seed=2), "b"))
    # Every marginal is unchanged...
    assert all(c.severity == "none" for c in report.columns)
    assert not report.column_drift
    # ...but the joint distribution is not.
    corr = report.correlation
    assert corr.drifted
    assert report.correlation_drift and report.dataset_drift
    assert report.status == "drift"
    top = corr.pairs[0]
    assert {top.column_a, top.column_b} == {"a", "b"}
    assert top.reference == pytest.approx(0.8, abs=0.05)
    assert abs(top.current) < 0.1
    assert top.change == pytest.approx(top.current - top.reference)
    assert top.exceeds_threshold
    assert [[p.column_a, p.column_b] for p in corr.flagged_pairs] == [["a", "b"]]
    # pairs are ordered by absolute change, largest first
    changes = [p.abs_change for p in corr.pairs]
    assert changes == sorted(changes, reverse=True)


def test_correlation_threshold_is_configurable():
    det = DriftDetector().fit(correlated_frame(3000, seed=1, rho=0.5))
    current = correlated_frame(3000, seed=2, rho=0.3)  # a moderate weakening of ~0.2
    shift = det.drift_report(current).correlation.max_abs_change
    assert 0.1 < shift < 0.3

    lenient = det.drift_report(current, thresholds=DriftThresholds(correlation_shift=shift + 0.05))
    strict = det.drift_report(current, thresholds={"drift": {"correlation_shift": shift - 0.05}})
    assert not lenient.correlation.drifted
    assert strict.correlation.drifted and strict.dataset_drift
    assert strict.correlation.threshold == pytest.approx(shift - 0.05)


def test_threshold_boundary_is_inclusive():
    profile = CorrelationProfile(columns=["a", "b"], matrix=[[1.0, 0.5], [0.5, 1.0]])
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    cur = pd.DataFrame({"a": a, "b": -a})  # r = -1, change = -1.5
    report = compare_correlations(profile, cur, threshold=1.5)
    assert report.pairs[0].change == pytest.approx(-1.5)
    assert report.drifted


def test_spearman_method():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.1, 5, 2000)
    ref = pd.DataFrame({"x": x, "y": np.exp(x)})  # monotonic but non-linear
    det = DriftDetector(correlation_method="spearman").fit(ref)
    assert det.correlation.method == "spearman"
    assert det.correlation.matrix[0][1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        DriftDetector(correlation_method="kendall-ish")


def test_missing_values_and_missing_columns():
    ref = correlated_frame(3000, seed=1)
    ref.loc[ref.index[:500], "a"] = np.nan  # pairwise-complete rows are used
    det = DriftDetector().fit(ref)
    assert not np.isnan(det.correlation.matrix[0][1])

    report = det.drift_report(correlated_frame(2000, seed=2).drop(columns=["c"]))
    assert report.correlation.columns == ["a", "b"]
    assert len(report.correlation.pairs) == 1

    # Only one numeric column left: no correlation check, but no crash either.
    assert det.drift_report(correlated_frame(500, seed=2)[["a", "cat"]]).correlation is None


def test_constant_or_sparse_columns_are_reported_as_undefined():
    ref = correlated_frame(3000, seed=1)
    det = DriftDetector().fit(ref)
    cur = correlated_frame(2000, seed=2)
    cur["c"] = 1.0  # constant: correlation undefined
    report = det.drift_report(cur)
    assert sorted(map(tuple, report.correlation.undefined_pairs)) == [("a", "c"), ("b", "c")]
    assert len(report.correlation.pairs) == 1

    sparse = correlated_frame(MIN_PAIR_ROWS - 1, seed=3)
    assert det.drift_report(sparse).correlation.pairs == []


def test_fewer_than_two_numeric_columns_gives_no_profile():
    df = pd.DataFrame({"v": np.arange(100.0), "c": ["a", "b"] * 50})
    assert DriftDetector().fit(df).correlation is None
    assert DriftDetector().fit(df).drift_report(df).correlation is None
    assert fit_correlation(df, ["v"]) is None


def test_non_string_column_names():
    rng = np.random.default_rng(0)
    a = rng.normal(size=1000)
    ref = pd.DataFrame({0: a, 1: a + rng.normal(scale=0.5, size=1000)})
    det = DriftDetector().fit(ref)
    assert det.correlation.columns == ["0", "1"]


def test_correlation_survives_serialisation(tmp_path):
    ref = correlated_frame(3000, seed=1)
    ref["const"] = 7.0  # NaN correlations must round-trip as strict JSON
    det = DriftDetector(correlation_method="spearman").fit(ref)
    path = tmp_path / "det.json"
    det.save(path)
    assert "NaN" not in path.read_text()
    loaded = DriftDetector.load(path)
    assert loaded.correlation_method == "spearman"
    assert loaded.correlation.columns == det.correlation.columns
    np.testing.assert_array_equal(np.array(loaded.correlation.matrix), np.array(det.correlation.matrix))

    cur = scrambled(correlated_frame(2000, seed=2), "b")
    assert loaded.drift_report(cur).correlation.to_dict() == det.drift_report(cur).correlation.to_dict()


def test_detector_files_without_correlation_still_load():
    det = DriftDetector().fit(correlated_frame(1000, seed=1))
    data = json.loads(det.to_json())
    del data["correlation"]
    del data["config"]["correlation_method"]
    del data["config"]["thresholds"]["correlation_shift"]
    old = DriftDetector.from_json(json.dumps(data))
    assert old.correlation is None
    assert old.thresholds.correlation_shift == DriftThresholds().correlation_shift
    report = old.drift_report(correlated_frame(1000, seed=2))
    assert report.correlation is None and report.status == "ok"


def test_report_dict_includes_correlation_section():
    det = DriftDetector().fit(correlated_frame(3000, seed=1))
    data = json.loads(det.drift_report(scrambled(correlated_frame(2000, seed=2), "b")).to_json())
    assert data["correlation_drift"] is True
    assert data["column_drift"] is False
    assert data["dataset_drift"] is True
    corr = data["correlation"]
    assert corr["drifted"] is True
    assert corr["flagged_pairs"] == [["a", "b"]]
    assert corr["n_pairs"] == 3
    assert set(corr["top_shifts"][0]) == {
        "column_a", "column_b", "reference", "current", "change", "abs_change", "exceeds_threshold",
    }


def test_invalid_correlation_threshold():
    with pytest.raises(ValueError):
        DriftThresholds(correlation_shift=0)
    with pytest.raises(ValueError):
        DriftThresholds(correlation_shift=2.5)
