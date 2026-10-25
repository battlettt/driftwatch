import json
import pickle
import subprocess
import sys

import pytest

from driftwatch import DriftDetector
from driftwatch.cli import main


@pytest.fixture
def csvs(tmp_path, reference_df, stable_df, drifted_df):
    paths = {}
    for name, df in {"ref": reference_df, "stable": stable_df, "drift": drifted_df}.items():
        paths[name] = tmp_path / f"{name}.csv"
        df.to_csv(paths[name], index=False)
    return paths


def test_fit_then_check_html(tmp_path, csvs, capsys):
    det_path = tmp_path / "detector.json"
    assert main(["fit", "--reference", str(csvs["ref"]), "--out", str(det_path), "--target", "y"]) == 0
    assert DriftDetector.load(det_path).target == "y"

    report_path = tmp_path / "report.html"
    rc = main(["check", "--detector", str(det_path), "--data", str(csvs["drift"]), "--report", str(report_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRIFT DETECTED" in out
    assert "x1" in out
    assert report_path.read_text().startswith("<!DOCTYPE html>")


def test_check_markdown_and_json_formats(tmp_path, csvs):
    det_path = tmp_path / "detector.json"
    main(["fit", "--reference", str(csvs["ref"]), "--out", str(det_path)])

    md_path = tmp_path / "out.txt"
    main(["check", "--detector", str(det_path), "--data", str(csvs["drift"]), "--report", str(md_path), "--format", "md"])
    assert md_path.read_text().startswith("# driftwatch report")

    inferred = tmp_path / "out.md"
    main(["check", "--detector", str(det_path), "--data", str(csvs["drift"]), "--report", str(inferred)])
    assert inferred.read_text().startswith("# driftwatch report")

    json_path = tmp_path / "out.json"
    main(["check", "--detector", str(det_path), "--data", str(csvs["stable"]), "--report", str(json_path)])
    assert json.loads(json_path.read_text())["status"] == "ok"


def test_fail_on_drift_exit_codes(tmp_path, csvs):
    det_path = tmp_path / "detector.json"
    main(["fit", "--reference", str(csvs["ref"]), "--out", str(det_path), "--target", "y"])
    base = ["check", "--detector", str(det_path), "--fail-on", "drift", "--data"]
    assert main(base + [str(csvs["stable"])]) == 0
    assert main(base + [str(csvs["drift"])]) == 1


def test_model_baseline_and_decay_via_cli(tmp_path, csvs, model, capsys):
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(pickle.dumps(model))
    det_path = tmp_path / "detector.json"
    rc = main(["fit", "--reference", str(csvs["ref"]), "--out", str(det_path), "--target", "y", "--model", str(model_path)])
    assert rc == 0
    assert "Model baseline" in capsys.readouterr().out

    rc = main(["check", "--detector", str(det_path), "--data", str(csvs["stable"]), "--model", str(model_path), "--fail-on", "decay"])
    assert rc == 0
    assert "Model performance: within tolerance" in capsys.readouterr().out


def test_cli_errors(tmp_path, csvs, capsys):
    assert main(["fit", "--reference", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "d.json")]) == 2
    assert "file not found" in capsys.readouterr().err
    bogus = tmp_path / "bogus.json"
    bogus.write_text('{"format": "nope"}')
    assert main(["check", "--detector", str(bogus), "--data", str(csvs["stable"])]) == 2


def test_python_dash_m_entry_point(tmp_path, csvs):
    det_path = tmp_path / "detector.json"
    report_path = tmp_path / "report.md"
    run = lambda *args: subprocess.run([sys.executable, "-m", "driftwatch", *args], capture_output=True, text=True)  # noqa: E731
    fit = run("fit", "--reference", str(csvs["ref"]), "--out", str(det_path), "--target", "y")
    assert fit.returncode == 0, fit.stderr
    check = run("check", "--detector", str(det_path), "--data", str(csvs["drift"]), "--report", str(report_path), "--format", "md")
    assert check.returncode == 0, check.stderr
    assert "DRIFT DETECTED" in check.stdout
    assert report_path.exists()


def test_check_with_config_overrides_stored_thresholds(tmp_path, csvs, capsys):
    det_path = tmp_path / "detector.json"
    main(["fit", "--reference", str(csvs["ref"]), "--out", str(det_path), "--target", "y"])
    base = ["check", "--detector", str(det_path), "--data", str(csvs["stable"]), "--fail-on", "drift"]
    assert main(base) == 0

    # absurdly strict PSI cut-offs turn the stable batch into "drift"
    strict = tmp_path / "strict.json"
    strict.write_text(json.dumps({"drift": {"psi_moderate": 0.00001, "psi_high": 0.00002}}))
    report_path = tmp_path / "r.json"
    assert main(base + ["--config", str(strict), "--report", str(report_path)]) == 1
    data = json.loads(report_path.read_text())
    assert data["thresholds"]["psi_high"] == 0.00002
    assert data["thresholds"]["correlation_shift"] == 0.2  # untouched keys keep the stored value
    assert "correlation" in data and data["correlation"]["n_pairs"] == 1
    capsys.readouterr()

    # the detector file itself is not modified by --config
    assert DriftDetector.load(det_path).thresholds.psi_high == 0.25


def test_check_config_errors_exit_2(tmp_path, csvs, capsys):
    det_path = tmp_path / "detector.json"
    main(["fit", "--reference", str(csvs["ref"]), "--out", str(det_path)])
    base = ["check", "--detector", str(det_path), "--data", str(csvs["stable"]), "--config"]
    assert main(base + [str(tmp_path / "nope.json")]) == 2
    assert "config file not found" in capsys.readouterr().err
    typo = tmp_path / "typo.json"
    typo.write_text('{"drift": {"psi_hihg": 0.3}}')
    assert main(base + [str(typo)]) == 2
    assert "unknown drift threshold" in capsys.readouterr().err


def test_fit_with_config_and_flag_precedence(tmp_path, csvs):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"drift": {"psi_high": 0.4, "correlation_shift": 0.3}, "decay": {"max_r2_drop": 0.1}}))
    det_path = tmp_path / "detector.json"
    rc = main([
        "fit", "--reference", str(csvs["ref"]), "--out", str(det_path),
        "--config", str(cfg), "--correlation-shift", "0.25", "--correlation-method", "spearman",
    ])
    assert rc == 0
    det = DriftDetector.load(det_path)
    assert det.thresholds.psi_high == 0.4  # from the config
    assert det.thresholds.correlation_shift == 0.25  # explicit flag wins
    assert det.thresholds.psi_moderate == 0.10  # default
    assert det.decay_thresholds.max_r2_drop == 0.1
    assert det.correlation.method == "spearman"


def test_check_prints_correlation_shifts(tmp_path, capsys):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    a = rng.normal(size=2000)
    ref = pd.DataFrame({"a": a, "b": a + rng.normal(scale=0.3, size=2000)})
    cur = ref.copy()
    cur["b"] = rng.permutation(cur["b"].to_numpy())
    ref.to_csv(tmp_path / "ref.csv", index=False)
    cur.to_csv(tmp_path / "cur.csv", index=False)
    det_path = tmp_path / "d.json"
    main(["fit", "--reference", str(tmp_path / "ref.csv"), "--out", str(det_path)])
    assert "Stored pearson correlations for 1 numeric column pairs." in capsys.readouterr().out
    rc = main(["check", "--detector", str(det_path), "--data", str(tmp_path / "cur.csv"), "--fail-on", "drift"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "Status: DRIFT DETECTED" in out
    assert "drifted columns: 0% of 2" in out
    assert "a x b:" in out and "beyond threshold" in out
