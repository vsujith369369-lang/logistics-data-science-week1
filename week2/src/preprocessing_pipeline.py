"""End-to-end orchestration of the Week 2 preprocessing pipeline."""
from pathlib import Path

import pandas as pd

from cleaning import apply_rules, deduplicate, enforce_schema, standardise_text
from features import derive_features
from imputation import impute
from ingest import load_raw
from outliers import treat_outliers
from profiling import defect_summary, profile
from validation import scorecard, validate

START, END = "2024-01-01", "2025-12-31"
FESTIVALS = ["2024-10-31", "2024-11-01", "2025-10-20", "2025-10-21"]


def run_pipeline(raw: pd.DataFrame) -> pd.DataFrame:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/quarantine").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    print(profile(raw).to_string())
    print(defect_summary(raw))
    raw_rows = len(raw)

    df = enforce_schema(raw)
    df = deduplicate(df)
    df = standardise_text(df)
    df, rejects = apply_rules(df, "data/quarantine/rule_violations.parquet")
    df = impute(df)
    df = derive_features(df, festival_dates=FESTIVALS)
    df = treat_outliers(df)
    df = impute(df)  # re-impute values nulled by outlier treatment

    report = validate(df, raw_rows, START, END)
    df.to_parquet("data/processed/deliveries_clean.parquet", index=False)
    report.to_csv("reports/validation_report.csv")
    scorecard(raw, df).to_csv("reports/quality_scorecard.csv", index=False)

    print(f"clean rows: {len(df):,} | quarantined: {len(rejects):,}")
    return df


if __name__ == "__main__":
    run_pipeline(load_raw("data/raw"))
