# driftwatch report

| | |
|---|---|
| **Status** | DRIFT DETECTED |
| **Overall drift score** | 0.394 (0 = identical, 1 = every column at/above PSI 0.25) |
| **Drifted columns (moderate or high)** | 4 of 10 (40%) |
| **Severity counts** | none: 4, low: 2, moderate: 1, high: 3 |
| **Reference rows** | 4,000 |
| **Current rows** | 3,000 |
| **Generated** | 2026-10-25 14:58:00 UTC |

## Top drifted columns

| Column | Type | Severity | PSI | What changed |
|---|---|---|---:|---|
| `purpose` | categorical | **high** | 1.130 | '__other__' 0.0% → 15.1%; 1 unseen category ('small_business') |
| `annual_income` | numeric | **high** | 0.360 | mean 61,034 → 48,176 (-0.45 σ); 0.2% outside reference range; missing 1.7% → 8.4% |
| `credit_score` | numeric | **high** | 0.269 | mean 691.1 → 660.0 (-0.56 σ); 0.2% outside reference range |
| `home_ownership` | categorical | **moderate** | 0.108 | 'RENT' 39.2% → 55.5% |
| `term_months` | numeric | **low** | 0.094 | mean 43.2 → 46.8 (+0.32 σ) |

## All columns

| Column | Type | Severity | PSI | JS div. | KS stat | KS p | χ² p | TV dist. | Missing (ref → cur) |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `purpose` | categorical | high | 1.130 | 0.081 | - | - | <0.0001 | 0.151 | 0.0% → 0.0% |
| `annual_income` | numeric | high | 0.360 | 0.062 | 0.207 | <0.0001 | - | - | 1.7% → 8.4% |
| `credit_score` | numeric | high | 0.269 | 0.047 | 0.218 | <0.0001 | - | - | 0.0% → 0.0% |
| `home_ownership` | categorical | moderate | 0.108 | 0.019 | - | - | <0.0001 | 0.162 | 0.0% → 0.0% |
| `term_months` | numeric | low | 0.094 | 0.017 | 0.148 | <0.0001 | - | - | 0.0% → 0.0% |
| `loan_amount` | numeric | low | 0.015 | 0.003 | 0.052 | 0.0002 | - | - | 0.0% → 0.0% |
| `age` | numeric | none | 0.008 | 0.001 | 0.022 | 0.3595 | - | - | 0.0% → 0.0% |
| `open_accounts` | numeric | none | 0.006 | 0.001 | 0.035 | 0.0331 | - | - | 0.0% → 0.0% |
| `employment_years` | numeric | none | 0.003 | 0.001 | 0.015 | 0.8401 | - | - | 0.0% → 0.0% |
| `region` | categorical | none | 0.002 | 0.000 | - | - | 0.3068 | 0.018 | 0.0% → 0.0% |

## Model performance

**Model performance has decayed** (accuracy, log_loss).

| Metric | Reference | Current | Change | Decayed |
|---|---:|---:|---:|---|
| accuracy | 0.7923 | 0.6457 | -0.1466 (-18.5%) | yes |
| f1 | 0.3069 | 0.6032 | +0.2963 (+96.5%) | no |
| log_loss | 0.4519 | 0.6436 | +0.1917 (+42.4%) | yes |

Decay rules: accuracy drop ≥ 0.05, F1 drop ≥ 0.05, log-loss increase ≥ 20%. Scored on 3,000 labelled rows.

## How to read this report

- **Severity** comes from the Population Stability Index: `high` if PSI ≥ 0.25, `moderate` if PSI ≥ 0.1, `low` if PSI is below 0.1 but the KS (numeric) or chi-square (categorical) test rejects equality at p < 0.01, otherwise `none`.
- The dataset is flagged as drifted when at least 30% of columns are `moderate` or worse, or any column is `high`.
- **JS div.** is the Jensen-Shannon divergence (base 2, 0 to 1) over the same bins as PSI. **TV dist.** is the share of probability mass that moved between categories.
