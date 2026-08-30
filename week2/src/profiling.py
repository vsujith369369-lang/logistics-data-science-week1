"""Data quality profiling: quantify every defect before repairing anything."""
import pandas as pd


def profile(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=True),
        "sample": [df[c].dropna().iloc[0] if df[c].notna().any() else None
                   for c in df.columns],
    })
    return report.sort_values("pct_missing", ascending=False)


def defect_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "duplicate_order_ids": int(df.duplicated(subset=["order_id"]).sum()),
        "total_null_cells": int(df.isna().sum().sum()),
        "carrier_variants": int(df["carrier"].nunique(dropna=True)),
    }
