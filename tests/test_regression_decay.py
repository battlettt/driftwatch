import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from driftwatch import DecayThresholds, DriftDetector, compare_performance, compute_metrics
from driftwatch.cli import main
from driftwatch.decay import compare_metrics, infer_task, regression_metrics


def make_regression_frame(n: int, seed: int, noise: float = 1.0, slope: float = 3.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.uniform(0, 10, n)
    y = slope * x1 + 0.5 * x2 + rng.normal(scale=noise, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def reg_reference():
    return make_regression_frame(3000, seed=1)


@pytest.fixture
def regressor(reg_reference):
    return LinearRegression().fit(reg_reference[["x1", "x2"]], reg_reference["y"])


def test_regression_metrics_match_sklearn_and_hand_computation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.0, 2.0, 4.0])
    m = regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx((0.5 + 0 + 1 + 0) / 4)
    assert m["rmse"] == pytest.approx(np.sqrt((0.25 + 0 + 1 + 0) / 4))
    assert m["r2"] == pytest.approx(1 - 1.25 / 5.0)
    assert m["n"] == 4
    with pytest.raises(ValueError):
        regression_metrics([1.0], [1.0])
    with pytest.raises(ValueError, match="NaN"):
        regression_metrics([1.0, 2.0], [np.nan, 1.0])


def test_task_is_inferred_from_the_model(regressor, model):
    assert infer_task(regressor) == "regression"
    assert infer_task(make_pipeline(StandardScaler(), LinearRegression())) == "regression"
    assert infer_task(model) == "classification"
    assert infer_task(object()) == "classification"


def test_compute_metrics_for_regressor(regressor, reg_reference):
    m = compute_metrics(regressor, reg_reference[["x1", "x2"]], reg_reference["y"])
    assert set(m) == {"r2", "mae", "rmse", "n"}
    assert m["r2"] > 0.85
    assert m["rmse"] == pytest.approx(1.0, abs=0.1)  # noise scale
    assert m["mae"] < m["rmse"]
    with pytest.raises(ValueError, match="task"):
        compute_metrics(regressor, reg_reference[["x1", "x2"]], reg_reference["y"], task="ranking")


def test_explicit_task_overrides_inference(reg_reference):
    class DuckRegressor:  # not a sklearn estimator, so inference would say classification
        def predict(self, X):
            return 3.0 * X["x1"] + 0.5 * X["x2"]

    m = compute_metrics(DuckRegressor(), reg_reference[["x1", "x2"]], reg_reference["y"], task="regression")
    assert set(m) == {"r2", "mae", "rmse", "n"}


def test_no_regression_decay_on_same_distribution(regressor, reg_reference):
    stable = make_regression_frame(2000, seed=2)
    report = compare_performance(
        regressor, reg_reference[["x1", "x2"]], reg_reference["y"], stable[["x1", "x2"]], stable["y"]
    )
    assert report.task == "regression"
    assert {c.metric for c in report.changes} == {"r2", "mae", "rmse"}
    assert not report.decayed


def test_regression_decay_when_relationship_changes(regressor, reg_reference):
    # Concept drift: x1's effect on the target halves and the noise doubles.
    changed = make_regression_frame(2000, seed=3, noise=2.0, slope=1.5)
    report = compare_performance(
        regressor, reg_reference[["x1", "x2"]], reg_reference["y"], changed[["x1", "x2"]], changed["y"]
    )
    assert report.decayed
    assert set(report.decayed_metrics) == {"r2", "mae", "rmse"}
    r2 = next(c for c in report.changes if c.metric == "r2")
    assert r2.change < -0.2
    rmse = next(c for c in report.changes if c.metric == "rmse")
    assert rmse.relative_change > 0.5


def test_regression_thresholds_are_respected():
    ref = {"r2": 0.80, "mae": 10.0, "rmse": 14.0, "n": 1000}
    cur = {"r2": 0.77, "mae": 10.8, "rmse": 15.75, "n": 500}
    report = compare_metrics(ref, cur)
    # r2 -0.03 < 0.05; mae +8% < 10%; rmse +12.5% >= 10%
    assert report.decayed_metrics == ["rmse"]
    assert report.rules == ["R² drop ≥ 0.05", "MAE increase ≥ 10%", "RMSE increase ≥ 10%"]
    strict = compare_metrics(ref, cur, DecayThresholds(max_r2_drop=0.02, max_mae_increase=0.05))
    assert strict.decayed_metrics == ["r2", "mae", "rmse"]
    lenient = compare_metrics(ref, cur, DecayThresholds(max_rmse_increase=0.2))
    assert not lenient.decayed


def test_negative_reference_r2_relative_change_has_the_right_sign():
    report = compare_metrics({"r2": -0.5}, {"r2": -1.0})
    (change,) = report.changes
    assert change.change == pytest.approx(-0.5)
    assert change.relative_change == pytest.approx(-1.0)  # worse, so negative
    assert change.decayed


def test_metrics_with_nothing_in_common_are_an_error():
    with pytest.raises(ValueError, match="no metrics in common"):
        compare_metrics({"accuracy": 0.9}, {"r2": 0.5})


def test_classifier_and_regressor_side_by_side(regressor, reg_reference, model, reference_df, stable_df):
    clf_report = compare_performance(
        model, reference_df[["x1", "x2"]], reference_df["y"], stable_df[["x1", "x2"]], stable_df["y"]
    )
    reg_report = compare_performance(
        regressor, reg_reference[["x1", "x2"]], reg_reference["y"],
        reg_reference[["x1", "x2"]], reg_reference["y"],
    )
    assert clf_report.task == "classification"
    assert reg_report.task == "regression"
    assert clf_report.to_dict()["task"] == "classification"
    assert reg_report.to_dict()["task"] == "regression"


def test_detector_with_regressor_baseline(tmp_path, reg_reference):
    model = RandomForestRegressor(n_estimators=30, random_state=0).fit(reg_reference[["x1", "x2"]], reg_reference["y"])
    det = DriftDetector().fit(reg_reference, target="y", model=model)
    assert set(det.baseline) == {"r2", "mae", "rmse", "n"}
    assert det.feature_names == ["x1", "x2"]

    path = tmp_path / "det.json"
    det.save(path)
    loaded = DriftDetector.load(path)

    changed = make_regression_frame(2000, seed=4, noise=3.0, slope=0.0)
    report = loaded.drift_report(changed, model=model)
    assert report.performance.decayed
    md = report.to_markdown()
    assert "Model performance has decayed" in md
    assert "| r2 |" in md and "| rmse |" in md
    assert "R² drop ≥ 0.05" in md and "accuracy drop" not in md
    html = report.to_html()
    assert "RMSE increase ≥ 10%" in html


def test_regression_decay_via_cli(tmp_path, regressor, reg_reference, capsys):
    model_path = tmp_path / "reg.pkl"
    model_path.write_bytes(pickle.dumps(regressor))
    ref_path, cur_path, det_path = tmp_path / "ref.csv", tmp_path / "cur.csv", tmp_path / "det.json"
    reg_reference.to_csv(ref_path, index=False)
    make_regression_frame(2000, seed=5, noise=2.5, slope=1.0).to_csv(cur_path, index=False)

    assert main(["fit", "--reference", str(ref_path), "--out", str(det_path), "--target", "y", "--model", str(model_path)]) == 0
    out = capsys.readouterr().out
    assert "Model baseline: r2=" in out and "rmse=" in out

    base = ["check", "--detector", str(det_path), "--data", str(cur_path), "--model", str(model_path), "--fail-on", "decay"]
    assert main(base) == 1
    assert "Model performance: DECAYED" in capsys.readouterr().out

    lenient = tmp_path / "lenient.json"
    lenient.write_text('{"decay": {"max_r2_drop": 5, "max_mae_increase": 10, "max_rmse_increase": 10}}')
    assert main(base + ["--config", str(lenient)]) == 0
    assert "Model performance: within tolerance" in capsys.readouterr().out
