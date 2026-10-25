# driftwatch

Detect data drift and model decay in tabular ML models before predictions quietly go bad.

driftwatch is a small Python library and CLI. You fit it once on the data your model was trained or validated on. After that it compares each new batch of production data against that reference, column by column and pair by pair. It tells you which features have shifted and how far, which feature relationships have broken, and whether model performance has dropped. It runs fully offline with numpy, pandas, scipy and scikit-learn only. It makes no network calls and needs no accounts or API keys.

## Why it matters

A model that scored well at launch can degrade without any error being raised. Typical causes:

- **Covariate drift.** The inputs move. A downturn lowers incomes, a new marketing channel brings in a different customer mix, or an upstream form change starts blanking a field.
- **Concept drift.** The relationship between inputs and outcome changes, so the same applicant profile is now riskier.

Labels often arrive weeks after predictions, so you can't just watch accuracy. driftwatch checks the **inputs** right away, using distribution-shift metrics that need no labels. It checks **performance** once labels arrive. Each check catches problems the other one misses.

## Features

- **Reference fitting.** `DriftDetector.fit(df)` stores the following for each column:
  - numeric columns: quantile bins, mean, std, range, missing rate, and a capped sample used for the KS test;
  - categorical columns: value frequencies, with `__other__` and `__missing__` buckets.

  It also stores the correlation matrix of the numeric columns. You can also store a performance baseline for a fitted scikit-learn classifier or regressor.
- **Per-column drift metrics:**
  - Population Stability Index (PSI), for all columns;
  - Jensen-Shannon divergence, for all columns;
  - Kolmogorov-Smirnov statistic and p-value, for numeric columns;
  - chi-square test and total-variation distance, for categorical columns.

  Each column gets a severity label (`none` / `low` / `moderate` / `high`).
- **Correlation shift (multivariate drift).** Compares the reference correlation matrix of the numeric columns against each new batch. It lists the pairs whose correlation changed most, with before and after values, and flags the dataset when any pair moves by more than a configurable amount. This catches broken relationships between features that per-column checks can't see.
- **Drift report.** It gives an overall drift score, the share of drifted columns, a dataset-level drift flag, per-column results and the top drifted columns. It also detects unseen categories, columns that went missing, and changes in missing-value rates.
- **Performance decay.** For classifiers it compares accuracy, F1 and log-loss on newly labelled data against the reference baseline. For regressors it compares R², MAE and RMSE. Decay beyond configurable tolerances is flagged.
- **Configurable thresholds.** All cut-offs (PSI, significance level, drift share, correlation shift, decay tolerances) can be loaded from a JSON config file, with `--config` on the CLI or `thresholds=` in the API.
- **Reports.** Markdown, JSON, and a self-contained HTML file that uses inline CSS only, with no fonts, scripts or CDN links.
- **CLI.** `fit` and `check` commands. The detector is saved as plain JSON. `--fail-on` turns drift or decay into a non-zero exit code, so a CI job or scheduled pipeline can stop on it.
- **Check history and trend.** `check --history PATH` appends a one-line JSON summary of each check to a JSONL file. `history` prints those checks as a table, and `trend` renders a self-contained HTML page charting the overall drift score over time.

## Install

Requires Python 3.9+. From the project root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # library + CLI
pip install -e '.[dev]'     # + pytest, for development
```

## Quickstart

### 1. Run the demo

```bash
python examples/loan_drift_demo.py
```

The demo builds a synthetic loan-default table and trains a logistic-regression pipeline on it. It then checks three batches against a reference slice:

- a **stable** batch drawn from the same distribution;
- a **drifted** batch that simulates a downturn plus a new acquisition channel: lower incomes and credit scores, more renters, a loan purpose the model has never seen, an upstream bug that blanks incomes, and a higher underlying default rate;
- a **scrambled** batch that simulates an upstream join bug: each application's loan amount is attached to the wrong row. Every column keeps exactly the same distribution, so only the correlation-shift check can notice.

Real output:

```text
=== Stable batch (same distribution) ===
status=ok  overall_score=0.017  drift_share=0%
  accuracy  0.7923 -> 0.8070
  f1        0.3069 -> 0.3164
  log_loss  0.4519 -> 0.4332
  correlation shift: max |change| = 0.036 (threshold 0.2)

=== Drifted batch (downturn + new channel) ===
status=drift  overall_score=0.394  drift_share=40%
  purpose           high      PSI=1.130  JS=0.081
  annual_income     high      PSI=0.360  JS=0.062
  credit_score      high      PSI=0.269  JS=0.047
  home_ownership    moderate  PSI=0.108  JS=0.019
  term_months       low       PSI=0.094  JS=0.017
  accuracy  0.7923 -> 0.6457  <-- decayed
  f1        0.3069 -> 0.6032
  log_loss  0.4519 -> 0.6436  <-- decayed
  correlation shift: max |change| = 0.027 (threshold 0.2)

=== Scrambled batch (join bug: loan_amount on the wrong rows) ===
status=drift  overall_score=0.016  drift_share=0%
  accuracy  0.7923 -> 0.7760
  f1        0.3069 -> 0.3101
  log_loss  0.4519 -> 0.4874
  correlation shift: max |change| = 0.722 (threshold 0.2)
    annual_income x loan_amount: +0.720 -> -0.001  <-- shifted
```

In the scrambled batch no column drifts (drift share 0%, overall score 0.016), yet the correlation between income and loan amount collapses from 0.72 to 0. Per-column metrics cannot catch this kind of failure.

The full report for the drifted batch is checked in as [`examples/sample-report.md`](examples/sample-report.md) and [`examples/sample-report.html`](examples/sample-report.html). The demo also writes CSVs and a pickled model to `examples/output/`, which is gitignored, so you can try the CLI on them.

F1 *rises* in the drifted batch even though accuracy and log-loss get much worse. Defaults are more common in that batch, so a model that flags some of them gets more true positives. A single metric can mislead you in this way, which is why driftwatch tracks several.

### 2. Use the CLI

```bash
python -m driftwatch fit \
    --reference examples/output/reference.csv \
    --target defaulted --model examples/output/model.pkl \
    --out detector.json
```

```text
Fitted detector on 4,000 rows: 7 numeric and 3 categorical columns.
Stored pearson correlations for 21 numeric column pairs.
Model baseline: accuracy=0.7923, f1=0.3069, log_loss=0.4519
Saved to detector.json
```

```bash
python -m driftwatch check \
    --detector detector.json \
    --data examples/output/current_drifted.csv \
    --model examples/output/model.pkl \
    --report report.html
```

```text
Status: DRIFT DETECTED
Overall drift score: 0.394 | drifted columns: 40% of 10
Top drifted columns:
  purpose         high      PSI=1.130  '__other__' 0.0% → 15.1%; 1 unseen category ('small_business')
  annual_income   high      PSI=0.360  mean 61,034 → 48,176 (-0.45 σ); 0.2% outside reference range; missing 1.7% → 8.4%
  credit_score    high      PSI=0.269  mean 691.1 → 660.0 (-0.56 σ); 0.2% outside reference range
  home_ownership  moderate  PSI=0.108  'RENT' 39.2% → 55.5%
  term_months     low       PSI=0.094  mean 43.2 → 46.8 (+0.32 σ)
Correlation shifts: No column pair changed correlation by 0.2 or more (largest change 0.027).
Model performance: DECAYED (accuracy, log_loss)
  accuracy  0.7923 -> 0.6457 (-0.1466)
  f1        0.3069 -> 0.6032 (+0.2963)
  log_loss  0.4519 -> 0.6436 (+0.1917)
Report written to report.html
```

The same check on the scrambled batch:

```bash
python -m driftwatch check --detector detector.json \
    --data examples/output/current_scrambled.csv --fail-on drift
```

```text
Status: DRIFT DETECTED
Overall drift score: 0.016 | drifted columns: 0% of 10
Correlation shifts: 1 column pair changed correlation by 0.2 or more (largest change 0.722).
  annual_income x loan_amount: +0.720 -> -0.001  <-- beyond threshold
```

It exits with status 1.

Useful options:

| Command | Option | Purpose |
|---|---|---|
| `fit` | `--target COL` | Label column. It is excluded from the drift checks and used for the model baseline. |
| `fit` | `--model PATH` | Pickled, fitted classifier or regressor. Stores accuracy / F1 / log-loss (classifier) or R² / MAE / RMSE (regressor) as the baseline. |
| `fit` | `--exclude COL ...` | Columns to ignore, such as IDs or timestamps. |
| `fit` | `--categorical COL ...` | Treat numeric codes as categories. |
| `fit` | `--bins`, `--max-categories`, `--sample-size` | Profile resolution. Defaults: 10, 50 and 5000. |
| `fit` | `--config PATH` | JSON threshold config (see [Configuring thresholds](#configuring-thresholds)). The values are stored in the detector JSON. |
| `fit` | `--psi-moderate`, `--psi-high`, `--p-value`, `--drift-share`, `--correlation-shift` | Override single drift thresholds. They take precedence over `--config` and are stored in the detector JSON. |
| `fit` | `--correlation-method {pearson,spearman}` | Correlation used for the correlation-shift check. Default: `pearson`. |
| `check` | `--config PATH` | JSON threshold config applied for this check only, on top of the thresholds stored in the detector. |
| `check` | `--format {html,md,json}` | Report format. By default it is inferred from the `--report` extension, or HTML if the extension is unknown. |
| `check` | `--model PATH` | Adds a performance-decay section (classifier or regressor). The data must contain the target column. |
| `check` | `--fail-on {drift,warning,decay}` | Exit with status 1 when the condition is met, for CI and cron jobs. |
| `check` | `--history PATH` | Append a one-line JSON summary of the check to a JSONL file (see [Tracking checks over time](#4-track-checks-over-time)). |
| `history` | `--last N`, `--newest-first` | Show only the N most recent checks; list newest first instead of oldest first. |
| `trend` | `--report PATH`, `--last N`, `--title TEXT` | Where to write the HTML page, how many recent checks to include, and the page title. |

Exit codes: `0` means the command ran (and the `--fail-on` condition, if set, was not met). `1` means the `--fail-on` condition was met. `2` means bad input, such as a missing file, an invalid detector, an invalid config file, or a malformed history file.

> `--model` loads a Python pickle. Unpickling can run arbitrary code, so only pass model files you created or otherwise trust.

### 3. Use the library

```python
import pandas as pd
from driftwatch import DriftDetector

reference = pd.read_csv("examples/output/reference.csv")
current = pd.read_csv("examples/output/current_drifted.csv")

detector = DriftDetector().fit(reference, target="defaulted")
report = detector.drift_report(current)

print(report.status, round(report.overall_score, 3))   # drift 0.394
for col in report.top_drifted:
    print(col.column, col.severity, round(col.psi, 3))

detector.save("detector.json")              # plain JSON, safe to commit or diff
same = DriftDetector.load("detector.json")

open("report.md", "w").write(report.to_markdown())
open("report.html", "w").write(report.to_html())
```

To include performance decay, pass a fitted classifier at fit time and again when checking:

```python
detector = DriftDetector().fit(reference, target="defaulted", model=model)
report = detector.drift_report(current, model=model)   # current must include the label column
report.performance.decayed            # True
report.performance.decayed_metrics    # ['accuracy', 'log_loss']
```

Regressors work the same way. driftwatch recognises a scikit-learn regressor, including a `Pipeline` that ends in one, and tracks R², MAE and RMSE instead:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from driftwatch import DriftDetector

rng = np.random.default_rng(0)
def batch(n, slope):
    x = rng.uniform(0, 10, n)
    return pd.DataFrame({"x": x, "price": slope * x + rng.normal(0, 1, n)})

reference, current = batch(2000, slope=3.0), batch(2000, slope=2.0)
model = LinearRegression().fit(reference[["x"]], reference["price"])

detector = DriftDetector().fit(reference, target="price", model=model)
detector.baseline                        # {'r2': 0.987, 'mae': 0.805, 'rmse': 0.997, 'n': 2000.0} (rounded)
perf = detector.drift_report(current, model=model).performance
print(perf.task, perf.decayed_metrics)   # regression ['r2', 'mae', 'rmse']
```

`compute_metrics` and `compare_performance` take an optional `task="regression"` (or `"classification"`) for models that aren't scikit-learn estimators.

Correlation shifts are on the report too:

```python
loans = DriftDetector().fit(pd.read_csv("examples/output/reference.csv"), target="defaulted")
report = loans.drift_report(pd.read_csv("examples/output/current_scrambled.csv"))
report.correlation_drift                 # True  (report.column_drift is False)
for pair in report.correlation.flagged_pairs:
    print(pair.column_a, pair.column_b, pair.reference, pair.current)
# annual_income loan_amount 0.720... -0.001...
```

You can also call the metrics and comparisons directly. See `driftwatch.psi`, `driftwatch.ks_test`, `driftwatch.js_divergence`, `driftwatch.chi_square_test`, `driftwatch.total_variation`, `driftwatch.compare_correlations` and `driftwatch.compare_performance`.

### 4. Track checks over time

A single check tells you about one batch. To see whether drift is creeping up across batches, pass `--history` to every `check`. Each check appends one JSON line to the file, which is created on first use. The line is written even when `--fail-on` makes the command exit with 1. These commands run the demo's three batches through the CLI:

```bash
python -m driftwatch fit --reference examples/output/reference.csv --target defaulted \
    --model examples/output/model.pkl --out examples/output/detector.json

for batch in stable drifted scrambled; do
  python -m driftwatch check --detector examples/output/detector.json \
      --data examples/output/current_$batch.csv --model examples/output/model.pkl \
      --history examples/output/history.jsonl
done
```

Each `check` prints its usual summary, followed by `Check appended to examples/output/history.jsonl`. One line of the file, pretty-printed:

```json
{
    "timestamp": "2026-10-25T15:17:30Z",
    "status": "ok",
    "overall_score": 0.016938,
    "drift_share": 0.0,
    "dataset_drift": false,
    "correlation_drift": false,
    "top_drifted": [],
    "max_correlation_shift": 0.035934,
    "model_decayed": false,
    "n_current": 3000,
    "missing_columns": [],
    "source": "examples/output/current_stable.csv",
    "driftwatch_version": "0.2.0"
}
```

`top_drifted` holds `{"column", "severity", "psi"}` objects. `model_decayed` is `null` when the check ran without `--model`. Timestamps are UTC, taken when the check ran.

`history` prints the recorded checks. By default they are listed **oldest first** (newest last), in the order they were appended. Rows are numbered by check order:

```bash
python -m driftwatch history --history examples/output/history.jsonl
```

```text
#  Time (UTC)           Status  Score  Drifted  Corr. drift  Decayed  Top drifted columns
1  2026-10-25 15:17:30  ok      0.017       0%  no           no       -
2  2026-10-25 15:17:32  drift   0.394      40%  no           yes      purpose, annual_income, credit_score, +2 more
3  2026-10-25 15:17:33  drift   0.016       0%  yes          no       -
```

`--newest-first` reverses the order, and `--last N` keeps only the N most recent checks. Rows keep their original numbers:

```bash
python -m driftwatch history --history examples/output/history.jsonl --newest-first --last 2
```

```text
#  Time (UTC)           Status  Score  Drifted  Corr. drift  Decayed  Top drifted columns
3  2026-10-25 15:17:33  drift   0.016       0%  yes          no       -
2  2026-10-25 15:17:32  drift   0.394      40%  no           yes      purpose, annual_income, credit_score, +2 more
```

The third check is a reminder that the score and the status measure different things. The score only reflects per-column PSI, so the scrambled batch scores 0.016. It is still `drift` because of the correlation-shift rule, which the `Corr. drift` column shows.

`trend` writes a self-contained HTML page. It shows the overall drift score over time as a step chart, with points placed by check time and coloured by status, and a table of the checks below it. Hovering a point shows its details. The page uses inline CSS and inline SVG only, with no scripts, fonts or network requests:

```bash
python -m driftwatch trend --history examples/output/history.jsonl --report examples/sample-trend.html
```

```text
Trend of 3 checks written to examples/sample-trend.html
```

The result is checked in as [`examples/sample-trend.html`](examples/sample-trend.html). In a real deployment the checks would be hours or days apart, not seconds.

In Python:

```python
from driftwatch import append_history, read_history, render_trend_html

append_history("history.jsonl", report, source="batch-2026-10-25.csv")   # returns the HistoryEntry
entries = read_history("history.jsonl")                                  # oldest first
print(entries[-1].status, entries[-1].overall_score)
open("trend.html", "w").write(render_trend_html(entries))
```

`read_history` skips blank lines and ignores keys it doesn't know. It rejects malformed lines with a `ValueError` that names the line number. Each append writes one line in append mode, so you can commit, `tail` or concatenate history files with ordinary tools. driftwatch does not lock the file, so give concurrent jobs separate history files.

## Configuring thresholds

Every threshold can be set from a JSON file. This is [`examples/thresholds.json`](examples/thresholds.json), which lists every available key. It tightens the `high` PSI cut-off to 0.20, the correlation-shift cut-off to 0.15, the accuracy tolerance to 0.03 and the log-loss tolerance to 10%:

```json
{
  "drift": {
    "psi_moderate": 0.1,
    "psi_high": 0.2,
    "p_value": 0.01,
    "drift_share": 0.3,
    "correlation_shift": 0.15
  },
  "decay": {
    "max_accuracy_drop": 0.03,
    "max_f1_drop": 0.05,
    "max_log_loss_increase": 0.1,
    "max_r2_drop": 0.05,
    "max_mae_increase": 0.1,
    "max_rmse_increase": 0.1
  }
}
```

| Section | Key | Default | Meaning |
|---|---|---|---|
| `drift` | `psi_moderate` | 0.10 | PSI at or above which a column is `moderate`. |
| `drift` | `psi_high` | 0.25 | PSI at or above which a column is `high`. Must be greater than `psi_moderate`. |
| `drift` | `p_value` | 0.01 | Significance level of the KS / chi-square tests, used for the `low` label. |
| `drift` | `drift_share` | 0.30 | Share of `moderate`-or-worse columns that flags the dataset. |
| `drift` | `correlation_shift` | 0.20 | Absolute change in a pair's correlation that flags the dataset. |
| `decay` | `max_accuracy_drop` | 0.05 | Absolute accuracy drop that counts as decay. |
| `decay` | `max_f1_drop` | 0.05 | Absolute F1 drop. |
| `decay` | `max_log_loss_increase` | 0.20 | Relative log-loss increase (0.20 = +20%). |
| `decay` | `max_r2_drop` | 0.05 | Absolute R² drop. |
| `decay` | `max_mae_increase` | 0.10 | Relative MAE increase. |
| `decay` | `max_rmse_increase` | 0.10 | Relative RMSE increase. |

Both sections and all keys are optional. A key you leave out keeps its current value: the value stored in the detector when you use `check --config`, or the default when you use `fit --config`. Unknown sections or keys, non-numeric values and out-of-range values are rejected with exit code 2, so a typo can't silently fall back to a default.

```bash
# store the thresholds in the detector at fit time
python -m driftwatch fit --reference examples/output/reference.csv --target defaulted \
    --config examples/thresholds.json --out detector.json

# or apply them to a single check, leaving the detector file unchanged
python -m driftwatch check --detector detector.json --data examples/output/current_drifted.csv \
    --config examples/thresholds.json
```

In Python, `thresholds=` accepts a config file path, a dict in the same shape, a `ThresholdConfig`, or a plain `DriftThresholds` / `DecayThresholds`:

```python
import pickle
import pandas as pd
from driftwatch import DriftDetector

reference = pd.read_csv("examples/output/reference.csv")
current = pd.read_csv("examples/output/current_drifted.csv")
with open("examples/output/model.pkl", "rb") as fh:   # trusted file written by the demo
    model = pickle.load(fh)

detector = DriftDetector(thresholds="examples/thresholds.json").fit(reference, target="defaulted", model=model)
detector.thresholds.psi_high                          # 0.2

# per-report override; the detector's stored thresholds are not changed
report = detector.drift_report(current, thresholds={"drift": {"correlation_shift": 0.5}})
decay = detector.performance_report(model, current, thresholds={"decay": {"max_accuracy_drop": 0.2}})
decay.decayed_metrics                                 # ['log_loss']
```

## Interpreting drift scores

### How each column is measured

| Metric | Applies to | What it measures | Range |
|---|---|---|---|
| **PSI** | all | `Σ (cur − ref) · ln(cur / ref)` over shared bins. Numeric bins are the reference deciles, plus a missing-value bin. Categorical bins are the categories, plus `__other__` and `__missing__`. | 0 to ∞ |
| **Jensen-Shannon divergence** | all | Symmetric, bounded distance over the same bins as PSI, computed with log base 2. | 0 to 1 |
| **KS statistic / p-value** | numeric | Largest gap between the two empirical CDFs, from a two-sample test against the stored reference sample. | 0 to 1 |
| **Chi-square p-value** | categorical | Two-sample test of homogeneity on reference vs current category counts. | 0 to 1 |
| **Total variation** | categorical | Share of probability mass that moved between categories. | 0 to 1 |

Empty bins are floored at 0.0001 before PSI takes logarithms, so PSI is always finite. A category that never appeared in the reference therefore produces a large but finite PSI. In the demo, `purpose` scores 1.13 because 15% of applications have the unseen purpose `small_business`.

### Severity labels

| Severity | Rule (defaults) | Meaning |
|---|---|---|
| `high` | PSI ≥ 0.25 | Significant shift. Act on it: investigate the pipeline, and consider retraining. |
| `moderate` | 0.10 ≤ PSI < 0.25 | Noticeable shift. Investigate. |
| `low` | PSI < 0.10 **and** KS / chi-square p < 0.01 | The change is statistically real but small. Keep watching. |
| `none` | otherwise | No meaningful change. |

The 0.10 and 0.25 PSI cut-offs are the long-established conventions from credit-risk model monitoring (e.g. Siddiqi, *Credit Risk Scorecards*, 2006): below 0.10 is stable, 0.10–0.25 is a moderate shift, and 0.25 or more is significant.

PSI alone drives the `moderate` and `high` labels on purpose. With large samples, hypothesis tests reject on trivially small differences. With tiny samples, they miss real ones. PSI measures the size of the shift rather than its statistical significance. The tests are used only to separate `low` (detectable) from `none`. They use p < 0.01 instead of 0.05 to limit false alarms when many columns are tested at once.

### Correlation shift

At fit time driftwatch stores the correlation matrix of all numeric columns. It uses Pearson by default, or Spearman with `correlation_method="spearman"` / `--correlation-method spearman` when relationships are monotonic but not linear. For each new batch it recomputes the matrix over the same columns and reports every pair's reference value, current value and change, largest change first. If any pair's correlation changes by at least `correlation_shift` (default 0.20) in absolute terms, the dataset is flagged as drifted.

The default of 0.20 is well outside sampling noise at typical batch sizes: with 1,000 rows, the standard error of a correlation coefficient is about 0.03. It is still small enough to catch a relationship that has weakened substantially, for example 0.65 → 0.45.

Correlations use pairwise-complete rows, meaning rows where both values are present. A pair is listed as "not compared" when either column is constant, or when fewer than 30 rows have both values present.

### Dataset-level results

- **Overall drift score** is the mean over columns of `min(PSI / 0.25, 1)`. A score of 0 means nothing moved. A score of 1 means every column is at or above the `high` threshold. A score of 0.4 means that, on average, columns have moved 40% of the way to "significant".
- **Drift share** is the fraction of columns rated `moderate` or `high`.
- **Status:**
  - `drift` when the drift share is ≥ 30%, any column is `high`, or any numeric column pair's correlation shifts by ≥ 0.20. The JSON report includes `column_drift` and `correlation_drift` so you can tell which rule fired;
  - `warning` when any column is `low` or above, or a reference column is missing from the new data;
  - `ok` otherwise.

### Performance decay

| Metric | Decay flagged when |
|---|---|
| accuracy | absolute drop ≥ 0.05 |
| F1 | absolute drop ≥ 0.05 (binary uses positive-class F1, where the positive class is `model.classes_[1]`; multiclass uses macro-F1) |
| log-loss | relative increase ≥ 20% (log-loss has no fixed scale, so the change is measured relative to the baseline) |
| R² (regression) | absolute drop ≥ 0.05 |
| MAE (regression) | relative increase ≥ 10% |
| RMSE (regression) | relative increase ≥ 10% |

Log-loss is only computed for models that have `predict_proba`. MAE and RMSE are in the units of the target, so they are compared as relative changes. R² is unit-free, so it is compared as an absolute change. The 10% regression defaults are reasonable starting points, not an established convention: set them to the error increase your use case can tolerate. All thresholds can be changed through `DriftThresholds` and `DecayThresholds`, or with a [config file](#configuring-thresholds).

### Caveats

- PSI depends on sample size. With a few hundred rows, sampling noise alone can reach PSI 0.05–0.1. Compare batches of a reasonable size, ideally 1,000+ rows.
- Only tracked categories are counted individually; by default that is the top 50. Rarer values, and values unseen in the reference, fall into `__other__`. They are still listed by name under "unseen categories".
- The correlation-shift check covers pairwise linear (Pearson) or monotonic (Spearman) relationships between numeric columns. It does not cover relationships involving categorical columns, non-monotonic dependence, or interactions among three or more columns. A change in those can still go unnoticed while every marginal distribution and pairwise correlation stays the same.
- The correlation matrix grows with the square of the number of numeric columns. With hundreds of numeric columns the detector JSON gets correspondingly larger; use `--exclude` for columns you don't need to monitor.

## Limitations

driftwatch is deliberately small. What it does **not** do:

- **Multivariate drift beyond pairwise correlation.** Each column is checked on its own, and relationships are checked only as pairwise Pearson or Spearman correlations between numeric columns. Interactions among three or more features, non-monotonic dependence, and relationships involving categorical columns are not monitored. A change confined to those can pass every check.
- **Label-free concept drift detection.** All the input checks measure changes in the distribution of the features (covariate drift). A change in the relationship between features and outcome, where the inputs look the same but the right answer is different, only shows up in the performance-decay check. That check needs labels for the new data, so concept drift is detected only after labels arrive.
- **Models other than classifiers and regressors.** Decay metrics exist only for classification (accuracy, F1, log-loss) and regression (R², MAE, RMSE). Ranking, clustering, forecasting-specific, multi-output and generative models are not supported.
- **Unstructured data.** Only tabular data (pandas DataFrames / CSV) is supported, not text, images or embeddings.
- **Scheduling, alerting and storage.** driftwatch runs one check when you call it. Running it on a schedule, sending alerts, and storing reports are left to your scheduler or CI. Check history is a local JSONL file without locking.

## Development

```bash
pip install -e '.[dev]'
pytest
```

The test suite covers:

- PSI edge cases: identical distributions, empty bins, invalid input;
- KS on identical vs shifted samples;
- categorical drift, unseen categories, and missing values;
- correlation shift: broken relationships with unchanged marginals, configurable cut-off, constant / sparse / missing columns, Spearman, and serialisation (including detector files written before the check existed);
- threshold config files: partial overrides, validation errors, and the API and CLI entry points;
- regression decay: R² / MAE / RMSE against hand-computed values, thresholds, detector baselines and the CLI;
- JSON round-trips;
- report escaping and self-containment;
- performance decay;
- check history: JSONL append / read round-trips, malformed lines, table ordering, trend-page self-containment, escaping and time-proportional placement, and the `check --history`, `history` and `trend` commands;
- the CLI, end to end on temporary CSV files, including `python -m driftwatch`.

CI (`.github/workflows/ci.yml`) runs the test suite and the demo script on Python 3.10, 3.11 and 3.12 for every push and pull request.

Project layout:

```text
src/driftwatch/
  metrics.py      PSI, JS divergence, KS, chi-square, total variation, binning
  thresholds.py   severity thresholds and rules
  config.py       JSON threshold config loading (ThresholdConfig)
  correlation.py  correlation-shift (multivariate drift) check
  detector.py     DriftDetector, DriftReport, ColumnDrift, JSON (de)serialisation
  decay.py        classifier and regressor performance metrics, decay comparison
  report.py       Markdown and HTML rendering
  history.py      check history (JSONL), history table, HTML trend page
  cli.py          `fit` / `check` / `history` / `trend` commands (also `python -m driftwatch`)
examples/
  loan_drift_demo.py   synthetic end-to-end demo
  sample-report.md     generated output, checked in
  sample-report.html   generated output, checked in
  sample-trend.html    generated trend page, checked in
  thresholds.json      example threshold config
tests/
.github/workflows/ci.yml
```

Release notes are in [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
