"""End-to-end driftwatch demo on a synthetic loan-default table.

What it does:

1. Generates a loan-application dataset where default risk depends on credit
   score, loan-to-income ratio, term, home ownership and purpose.
2. Trains a logistic-regression pipeline on one slice of it.
3. Fits a DriftDetector (with a model performance baseline) on a held-out
   reference slice.
4. Checks two "production" batches against it:
   * a stable batch drawn from the same distribution (should come back clean),
   * a drifted batch simulating an economic downturn plus a new marketing
     channel: lower incomes, lower credit scores, bigger loans, more renters,
     a loan purpose the model never saw, an upstream bug that blanks some
     incomes, and a higher underlying default rate,
   * a "scrambled" batch simulating an upstream join bug that attaches each
     application's loan amount to the wrong row. Every column keeps its
     distribution, so per-column checks see nothing; only the correlation
     shift check notices that loan amount no longer tracks income.
5. Prints the summaries and writes examples/sample-report.md and
   examples/sample-report.html for the drifted batch. CSVs and the pickled
   model go to examples/output/ so the CLI can be tried on them.

Run from the repository root:

    python examples/loan_drift_demo.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from driftwatch import DriftDetector

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

NUMERIC = ["age", "annual_income", "credit_score", "loan_amount", "term_months", "employment_years", "open_accounts"]
CATEGORICAL = ["home_ownership", "purpose", "region"]
TARGET = "defaulted"


def make_loans(n: int, seed: int, drifted: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n)
    income = rng.lognormal(mean=10.9, sigma=0.45, size=n)
    credit = rng.normal(690, 55, n)
    loan_ratio = rng.uniform(0.10, 0.50, n)
    term_p = [0.7, 0.3]
    home_p = [0.40, 0.45, 0.15]
    purposes = ["debt_consolidation", "credit_card", "home_improvement", "car", "other"]
    purpose_p = [0.45, 0.25, 0.12, 0.10, 0.08]

    if drifted:
        income = income * 0.8
        credit = rng.normal(660, 65, n)
        loan_ratio = rng.uniform(0.15, 0.65, n)
        term_p = [0.55, 0.45]
        home_p = [0.55, 0.33, 0.12]
        purposes = purposes + ["small_business"]
        purpose_p = [0.38, 0.22, 0.10, 0.08, 0.07, 0.15]

    credit = np.clip(credit, 300, 850).round()
    loan = np.clip((income * loan_ratio).round(-2), 1000, 60000)
    term = rng.choice([36, 60], size=n, p=term_p)
    home = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n, p=home_p)
    purpose = rng.choice(purposes, size=n, p=purpose_p)
    region = rng.choice(["north", "south", "east", "west"], size=n, p=[0.25, 0.30, 0.20, 0.25])
    employment = np.clip(rng.gamma(2.0, 3.0, n), 0, 45).round(1)
    open_accounts = rng.poisson(6, n)

    logit = (
        -1.9
        - 0.018 * (credit - 690)
        + 4.0 * (loan / income - 0.3)
        + 0.5 * (term == 60)
        + 0.4 * (home == "RENT")
        - 0.04 * (employment - 6)
        + 0.6 * (purpose == "small_business")
    )
    if drifted:
        # Macro shock the model has no feature for: everyone is riskier.
        logit = logit + 0.9
    defaulted = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "age": age,
            "annual_income": income.round(-2),
            "credit_score": credit.astype(int),
            "loan_amount": loan,
            "term_months": term,
            "employment_years": employment,
            "open_accounts": open_accounts,
            "home_ownership": home,
            "purpose": purpose,
            "region": region,
            TARGET: defaulted.astype(int),
        }
    )
    # Income is self-reported and occasionally missing; in the drifted batch an
    # upstream form change blanks it far more often.
    missing_rate = 0.08 if drifted else 0.02
    df.loc[rng.random(n) < missing_rate, "annual_income"] = np.nan
    return df


def train_model(train: pd.DataFrame) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    model = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(train[NUMERIC + CATEGORICAL], train[TARGET])
    return model


def print_summary(title: str, report) -> None:
    print(f"\n=== {title} ===")
    print(f"status={report.status}  overall_score={report.overall_score:.3f}  drift_share={report.drift_share:.0%}")
    for col in report.top_drifted:
        print(f"  {col.column:<17} {col.severity:<9} PSI={col.psi:.3f}  JS={col.js_divergence:.3f}")
    if report.performance is not None:
        for ch in report.performance.changes:
            flag = "  <-- decayed" if ch.decayed else ""
            print(f"  {ch.metric:<9} {ch.reference:.4f} -> {ch.current:.4f}{flag}")
    corr = report.correlation
    if corr is not None:
        print(f"  correlation shift: max |change| = {corr.max_abs_change:.3f} (threshold {corr.threshold})")
        for pair in corr.flagged_pairs:
            print(f"    {pair.column_a} x {pair.column_b}: {pair.reference:+.3f} -> {pair.current:+.3f}  <-- shifted")


def main() -> None:
    train = make_loans(6000, seed=1)
    reference = make_loans(4000, seed=2)
    stable = make_loans(3000, seed=3)
    drifted = make_loans(3000, seed=4, drifted=True)
    scrambled = make_loans(3000, seed=5)
    scrambled["loan_amount"] = np.random.default_rng(5).permutation(scrambled["loan_amount"].to_numpy())

    model = train_model(train)
    detector = DriftDetector().fit(reference, target=TARGET, model=model)

    stable_report = detector.drift_report(stable, model=model)
    drifted_report = detector.drift_report(drifted, model=model)
    print_summary("Stable batch (same distribution)", stable_report)
    print_summary("Drifted batch (downturn + new channel)", drifted_report)
    print_summary("Scrambled batch (join bug: loan_amount on the wrong rows)", detector.drift_report(scrambled, model=model))

    (HERE / "sample-report.md").write_text(drifted_report.to_markdown(), encoding="utf-8")
    (HERE / "sample-report.html").write_text(drifted_report.to_html(), encoding="utf-8")

    OUTPUT.mkdir(exist_ok=True)
    reference.to_csv(OUTPUT / "reference.csv", index=False)
    stable.to_csv(OUTPUT / "current_stable.csv", index=False)
    drifted.to_csv(OUTPUT / "current_drifted.csv", index=False)
    scrambled.to_csv(OUTPUT / "current_scrambled.csv", index=False)
    with open(OUTPUT / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    detector.save(OUTPUT / "detector.json")

    print(f"\nWrote {HERE / 'sample-report.md'} and {HERE / 'sample-report.html'}")
    print(f"Wrote CSVs, model.pkl and detector.json to {OUTPUT}")


if __name__ == "__main__":
    main()
