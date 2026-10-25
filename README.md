# driftwatch

Detect data drift and model decay in tabular ML models before predictions quietly go bad.

driftwatch is a small Python library and CLI. You fit it once on the data your model was trained or validated on. After that it compares each new batch of production data against that reference, column by column, and tells you which features have shifted, how far, and whether model accuracy has dropped. It runs fully offline with numpy, pandas, scipy and scikit-learn only. It makes no network calls and needs no accounts or API keys.

## Why it matters

A model that scored well at launch can degrade without any error being raised. Typical causes:

- **Covariate drift.** The inputs move. A downturn lowers incomes, a new marketing channel brings in a different customer mix, or an upstream form change starts blanking a field.
- **Concept drift.** The relationship between inputs and outcome changes, so the same applicant profile is now riskier.

Labels often arrive weeks after predictions, so you can't just watch accuracy. driftwatch checks the **inputs** right away, using distribution-shift metrics that need no labels. It checks **performance** once labels arrive. Each check catches problems the other one misses.

## Features

- **Reference fitting.** `DriftDetector.fit(df)` stores the following for each column:
  - numeric columns: quantile bins, mean, std, range, missing rate, and a capped sample used for the KS test;
  - categorical columns: value frequencies, with `__other__` and `__missing__` buckets.

  You can also store a performance baseline for a fitted scikit-learn classifier.
- **Per-column drift metrics:**
  - Population Stability Index (PSI), for all columns;
  - Jensen-Shannon divergence, for all columns;
  - Kolmogorov-Smirnov statistic and p-value, for numeric columns;
  - chi-square test and total-variation distance, for categorical columns.

  Each column gets a severity label (`none` / `low` / `moderate` / `high`).
- **Drift report.** It gives an overall drift score, the share of drifted columns, a dataset-level drift flag, per-column results and the top drifted columns. It also detects unseen categories, columns that went missing, and changes in missing-value rates.
- **Performance decay.** Compares accuracy, F1 and log-loss on newly labelled data against the reference baseline, and flags decay beyond configurable thresholds.
- **Reports.** Markdown, JSON, and a self-contained HTML file that uses inline CSS only, with no fonts, scripts or CDN links.
- **CLI.** `fit` and `check` commands. The detector is saved as plain JSON. `--fail-on` turns drift or decay into a non-zero exit code, so a CI job or scheduled pipeline can stop on it.

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

The demo builds a synthetic loan-default table and trains a logistic-regression pipeline on it. It then checks two batches against a reference slice:

- a **stable** batch drawn from the same distribution;
- a **drifted** batch that simulates a downturn plus a new acquisition channel: lower incomes and credit scores, more renters, a loan purpose the model has never seen, an upstream bug that blanks incomes, and a higher underlying default rate.

Real output:

```text
=== Stable batch (same distribution) ===
status=ok  overall_score=0.017  drift_share=0%
  accuracy  0.7923 -> 0.8070
  f1        0.3069 -> 0.3164
  log_loss  0.4519 -> 0.4332

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
```

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
Model performance: DECAYED (accuracy, log_loss)
  accuracy  0.7923 -> 0.6457 (-0.1466)
  f1        0.3069 -> 0.6032 (+0.2963)
  log_loss  0.4519 -> 0.6436 (+0.1917)
Report written to report.html
```

Useful options:

| Command | Option | Purpose |
|---|---|---|
| `fit` | `--target COL` | Label column. It is excluded from the drift checks and used for the model baseline. |
| `fit` | `--model PATH` | Pickled, fitted classifier. Stores accuracy / F1 / log-loss as the baseline. |
| `fit` | `--exclude COL ...` | Columns to ignore, such as IDs or timestamps. |
| `fit` | `--categorical COL ...` | Treat numeric codes as categories. |
| `fit` | `--bins`, `--max-categories`, `--sample-size` | Profile resolution. Defaults: 10, 50 and 5000. |
| `fit` | `--psi-moderate`, `--psi-high`, `--p-value`, `--drift-share` | Override the severity thresholds. They are stored in the detector JSON. |
| `check` | `--format {html,md,json}` | Report format. By default it is inferred from the `--report` extension, or HTML if the extension is unknown. |
| `check` | `--model PATH` | Adds a performance-decay section. The data must contain the target column. |
| `check` | `--fail-on {drift,warning,decay}` | Exit with status 1 when the condition is met, for CI and cron jobs. |

Exit codes: `0` means the check ran (and the `--fail-on` condition, if set, was not met). `1` means the `--fail-on` condition was met. `2` means bad input, such as a missing file or an invalid detector.

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

You can also call the metrics and the decay comparison directly. See `driftwatch.psi`, `driftwatch.ks_test`, `driftwatch.js_divergence`, `driftwatch.chi_square_test`, `driftwatch.total_variation` and `driftwatch.compare_performance`.

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

### Dataset-level results

- **Overall drift score** is the mean over columns of `min(PSI / 0.25, 1)`. A score of 0 means nothing moved. A score of 1 means every column is at or above the `high` threshold. A score of 0.4 means that, on average, columns have moved 40% of the way to "significant".
- **Drift share** is the fraction of columns rated `moderate` or `high`.
- **Status:**
  - `drift` when the drift share is ≥ 30%, or any column is `high`;
  - `warning` when any column is `low` or above, or a reference column is missing from the new data;
  - `ok` otherwise.

### Performance decay

| Metric | Decay flagged when |
|---|---|
| accuracy | absolute drop ≥ 0.05 |
| F1 | absolute drop ≥ 0.05 (binary uses positive-class F1, where the positive class is `model.classes_[1]`; multiclass uses macro-F1) |
| log-loss | relative increase ≥ 20% (log-loss has no fixed scale, so the change is measured relative to the baseline) |

Log-loss is only computed for models that have `predict_proba`. All thresholds can be changed through `DriftThresholds` and `DecayThresholds`.

### Caveats

- PSI depends on sample size. With a few hundred rows, sampling noise alone can reach PSI 0.05–0.1. Compare batches of a reasonable size, ideally 1,000+ rows.
- Only tracked categories are counted individually; by default that is the top 50. Rarer values, and values unseen in the reference, fall into `__other__`. They are still listed by name under "unseen categories".
- Each column is checked independently. driftwatch does not detect a change in the joint distribution while every marginal distribution stays the same.

## Development

```bash
pip install -e '.[dev]'
pytest
```

The test suite covers:

- PSI edge cases: identical distributions, empty bins, invalid input;
- KS on identical vs shifted samples;
- categorical drift, unseen categories, and missing values;
- JSON round-trips;
- report escaping and self-containment;
- performance decay;
- the CLI, end to end on temporary CSV files, including `python -m driftwatch`.

Project layout:

```text
src/driftwatch/
  metrics.py      PSI, JS divergence, KS, chi-square, total variation, binning
  thresholds.py   severity thresholds and rules
  detector.py     DriftDetector, DriftReport, ColumnDrift, JSON (de)serialisation
  decay.py        performance metrics and decay comparison
  report.py       Markdown and HTML rendering
  cli.py          `fit` / `check` commands (also `python -m driftwatch`)
examples/
  loan_drift_demo.py   synthetic end-to-end demo
  sample-report.md     generated output, checked in
  sample-report.html   generated output, checked in
tests/
```

## License

MIT. See [LICENSE](LICENSE).
