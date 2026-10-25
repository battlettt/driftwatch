import numpy as np
import pandas as pd
import pytest
from sklearn.svm import LinearSVC

from driftwatch import DecayThresholds, DriftDetector, compare_performance, compute_metrics
from driftwatch.decay import compare_metrics

from .conftest import make_frame


def _flip_labels(df: pd.DataFrame, share: float, seed: int = 0) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    idx = rng.random(len(out)) < share
    out.loc[idx, "y"] = 1 - out.loc[idx, "y"]
    return out


def test_compute_metrics(model, reference_df):
    m = compute_metrics(model, reference_df[["x1", "x2"]], reference_df["y"])
    assert set(m) == {"accuracy", "f1", "log_loss", "n"}
    assert 0.6 < m["accuracy"] <= 1
    assert m["n"] == len(reference_df)


def test_no_decay_on_same_distribution(model, reference_df, stable_df):
    report = compare_performance(
        model, reference_df[["x1", "x2"]], reference_df["y"], stable_df[["x1", "x2"]], stable_df["y"]
    )
    assert not report.decayed
    assert report.decayed_metrics == []
    assert {c.metric for c in report.changes} == {"accuracy", "f1", "log_loss"}


def test_decay_detected_when_labels_change(model, reference_df, stable_df):
    # Concept drift: same inputs, but the label relationship has changed for 30% of rows.
    changed = _flip_labels(stable_df, 0.3)
    report = compare_performance(
        model, reference_df[["x1", "x2"]], reference_df["y"], changed[["x1", "x2"]], changed["y"]
    )
    assert report.decayed
    assert "accuracy" in report.decayed_metrics
    assert "log_loss" in report.decayed_metrics
    acc = next(c for c in report.changes if c.metric == "accuracy")
    assert acc.change < -0.1


def test_decay_thresholds_are_respected():
    ref = {"accuracy": 0.90, "f1": 0.80, "log_loss": 0.30}
    cur = {"accuracy": 0.87, "f1": 0.74, "log_loss": 0.35}
    report = compare_metrics(ref, cur)
    assert report.decayed_metrics == ["f1"]  # acc -0.03 < 0.05; log-loss +16.7% < 20%
    lenient = compare_metrics(ref, cur, DecayThresholds(max_f1_drop=0.1))
    assert not lenient.decayed
    strict = compare_metrics(ref, cur, DecayThresholds(max_accuracy_drop=0.02, max_log_loss_increase=0.1))
    assert strict.decayed_metrics == ["accuracy", "f1", "log_loss"]


def test_model_without_predict_proba_skips_log_loss(reference_df):
    svc = LinearSVC().fit(reference_df[["x1", "x2"]], reference_df["y"])
    m = compute_metrics(svc, reference_df[["x1", "x2"]], reference_df["y"])
    assert "log_loss" not in m
    report = compare_metrics(m, m)
    assert [c.metric for c in report.changes] == ["accuracy", "f1"]


def test_multiclass_uses_macro_f1():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=600)})
    y = np.digitize(X["a"], [-0.5, 0.5])  # 3 classes
    clf = LogisticRegression().fit(X, y)
    m = compute_metrics(clf, X, y)
    assert m["f1"] > 0.7 and "log_loss" in m


def test_detector_baseline_and_decay(reference_df, model):
    det = DriftDetector().fit(reference_df, target="y", model=model)
    assert det.baseline["accuracy"] > 0.6
    assert det.feature_names == ["x1", "x2"]

    # Covariate shift that also breaks the model: labels follow a different rule.
    shifted = make_frame(2000, seed=7, shift=1.0)
    shifted = _flip_labels(shifted, 0.35, seed=1)
    report = det.drift_report(shifted, model=model)
    assert report.performance is not None
    assert report.performance.decayed
    assert "Model performance has decayed" in report.to_markdown()


def test_performance_requires_baseline(reference_df, stable_df, model):
    det = DriftDetector().fit(reference_df, target="y")
    with pytest.raises(ValueError, match="baseline"):
        det.performance_report(model, stable_df)
    with pytest.raises(ValueError, match="target"):
        DriftDetector().fit(reference_df, model=model)
