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
