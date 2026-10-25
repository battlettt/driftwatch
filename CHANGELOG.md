# Changelog

All notable changes to driftwatch are recorded here. Versions follow [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-25

### Added

- Correlation-shift check: compares the reference Pearson or Spearman correlation matrix of numeric columns with each new batch and flags pairs whose correlation moves by `correlation_shift` (default 0.20) or more. `fit --correlation-method` selects the method.
- JSON threshold config files for drift and decay thresholds (`fit --config`, `check --config`, `thresholds=` in the API), plus `fit` flags that override single drift thresholds.
- Regression decay: R², MAE and RMSE against a stored baseline, with configurable tolerances.
- GitHub Actions workflow that runs the tests and the demo script on Python 3.10, 3.11 and 3.12.
- Check history: `check --history PATH` appends a one-line JSON summary of each check to a JSONL file (`append_history` / `read_history` in the API).
- `driftwatch history` prints a table of recorded checks; `driftwatch trend` writes a self-contained HTML page charting the overall drift score over time.

## [0.1.0] - 2026-10-25

Initial release.

### Added

- `DriftDetector` that fits per-column reference profiles and scores new data with PSI, Jensen-Shannon divergence, KS (numeric) and chi-square / total variation (categorical), with severity labels, an overall drift score and a dataset-level drift flag.
- Classifier performance decay: accuracy, F1 and log-loss against a stored baseline.
- CLI with `fit` and `check` commands and `--fail-on` exit codes.
- Markdown, JSON and self-contained HTML reports.
