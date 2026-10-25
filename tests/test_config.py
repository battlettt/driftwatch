import json
from pathlib import Path

import pytest

from driftwatch import DecayThresholds, DriftDetector, DriftThresholds, ThresholdConfig
from driftwatch.config import resolve_thresholds

EXAMPLE_CONFIG = Path(__file__).resolve().parent.parent / "examples" / "thresholds.json"


def test_example_config_from_readme_loads():
    cfg = ThresholdConfig.load(EXAMPLE_CONFIG)
    raw = json.loads(EXAMPLE_CONFIG.read_text())
    assert cfg.drift.to_dict() == raw["drift"]
    assert cfg.decay.to_dict() == raw["decay"]
    # every threshold is documented by the example file
    assert set(raw["drift"]) == set(DriftThresholds().to_dict())
    assert set(raw["decay"]) == set(DecayThresholds().to_dict())


def test_partial_config_keeps_other_values():
    base = ThresholdConfig(DriftThresholds(psi_high=0.4), DecayThresholds(max_f1_drop=0.2))
    cfg = ThresholdConfig.from_dict({"drift": {"correlation_shift": 0.1}}, base)
    assert cfg.drift.correlation_shift == 0.1
    assert cfg.drift.psi_high == 0.4
    assert cfg.decay == base.decay
    assert ThresholdConfig.from_dict({}) == ThresholdConfig()


@pytest.mark.parametrize(
    "data, message",
    [
        ({"drfit": {}}, "unknown threshold config section"),
        ({"drift": {"psi_hgh": 0.3}}, "unknown drift threshold"),
        ({"decay": {"max_r2_drop": "0.1"}}, "must be a number"),
        ({"decay": {"max_r2_drop": True}}, "must be a number"),
        ({"drift": [0.1]}, "must be a JSON object"),
        ({"drift": {"psi_moderate": 0.5, "psi_high": 0.25}}, "psi_moderate < psi_high"),
        ({"decay": {"max_mae_increase": -0.1}}, "non-negative"),
        ([], "must be a JSON object"),
    ],
)
def test_invalid_configs_are_rejected(data, message):
    with pytest.raises(ValueError, match=message):
        ThresholdConfig.from_dict(data)


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="config file not found"):
        ThresholdConfig.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        ThresholdConfig.load(bad)


def test_resolve_thresholds_accepts_every_spec(tmp_path):
    base = ThresholdConfig()
    drift = DriftThresholds(p_value=0.05)
    decay = DecayThresholds(max_r2_drop=0.2)
    assert resolve_thresholds(None, base) is base
    assert resolve_thresholds(drift, base) == ThresholdConfig(drift, base.decay)
    assert resolve_thresholds(decay, base) == ThresholdConfig(base.drift, decay)
    full = ThresholdConfig(drift, decay)
    assert resolve_thresholds(full, base) is full
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"decay": {"max_r2_drop": 0.2}}))
    assert resolve_thresholds(path, base).decay.max_r2_drop == 0.2
    assert resolve_thresholds(str(path), base).decay.max_r2_drop == 0.2
    with pytest.raises(TypeError):
        resolve_thresholds(0.1, base)


def test_detector_accepts_config_path_and_stores_it(tmp_path, reference_df):
    det = DriftDetector(thresholds=EXAMPLE_CONFIG).fit(reference_df, target="y")
    assert det.thresholds.psi_high == 0.2
    assert det.thresholds.correlation_shift == 0.15
    assert det.decay_thresholds.max_accuracy_drop == 0.03
    path = tmp_path / "det.json"
    det.save(path)
    loaded = DriftDetector.load(path)
    assert loaded.thresholds == det.thresholds
    assert loaded.decay_thresholds == det.decay_thresholds

    # An explicit decay_thresholds argument wins over the config's decay section.
    explicit = DecayThresholds(max_accuracy_drop=0.5)
    assert DriftDetector(thresholds=EXAMPLE_CONFIG, decay_thresholds=explicit).decay_thresholds == explicit


def test_report_level_override_does_not_mutate_detector(reference_df, stable_df):
    det = DriftDetector().fit(reference_df, target="y")
    lowest = min(c.psi for c in det.drift_report(stable_df).columns)
    override = {"drift": {"psi_moderate": lowest / 4, "psi_high": lowest / 2}}
    report = det.drift_report(stable_df, thresholds=override)
    assert report.status == "drift"
    assert all(c.severity == "high" for c in report.columns)
    assert report.thresholds.psi_high == lowest / 2
    # the detector's own thresholds are untouched
    assert det.thresholds == DriftThresholds()
    assert det.drift_report(stable_df).status == "ok"


def test_report_level_decay_override(reference_df, stable_df, model):
    det = DriftDetector().fit(reference_df, target="y", model=model)
    broken = stable_df.copy()
    broken["y"] = 1 - broken["y"]  # every label inverted: the model is now badly wrong
    assert det.drift_report(broken, model=model).performance.decayed

    lenient = {"decay": {"max_accuracy_drop": 1.0, "max_f1_drop": 1.0, "max_log_loss_increase": 100.0}}
    assert not det.drift_report(broken, model=model, thresholds=lenient).performance.decayed
    assert not det.performance_report(model, broken, thresholds=lenient).decayed
    assert not det.performance_report(model, broken, thresholds=DecayThresholds(1.0, 1.0, 100.0)).decayed
    # the stored thresholds are untouched
    assert det.decay_thresholds == DecayThresholds()
