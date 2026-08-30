"""Phase 1-2: data collection, cleaning and validation.

Week 1 Task - Strategic Planning and Data Exploration in Logistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "order_id", "sku", "quantity", "qty_shipped", "weight_kg",
    "order_date", "promised_date", "delivered_date", "lat", "lon", "dc",
]


def load_orders(path: str) -> pd.DataFrame:
    """Load raw order lines and apply the cleaning rules from the report."""
    df = pd.read_csv(
        path,
        parse_dates=["order_date", "promised_date", "delivered_date"],
    )
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # 1. de-duplicate at the order-line grain
    df = df.drop_duplicates(subset=["order_id", "sku"])

    # 2. drop invalid quantities (returns are handled in a separate feed)
    df = df[df["quantity"] > 0]

    # 3. impute missing weights with the SKU median
    df["weight_kg"] = df.groupby("sku")["weight_kg"].transform(
        lambda s: s.fillna(s.median())
    )

    # 4. rows without geocode or delivery timestamp cannot be modelled
    df = df.dropna(subset=["lat", "lon", "delivered_date"])

    # 5. derived target fields
    df["delay_min"] = (
        df["delivered_date"] - df["promised_date"]
    ).dt.total_seconds() / 60
    df["is_late"] = (df["delay_min"] > 0).astype(int)
    df["dispatch_hour"] = df["order_date"].dt.hour
    df["is_weekend"] = df["order_date"].dt.dayofweek.isin([5, 6]).astype(int)

    # 6. cap extreme durations at the 99th percentile
    cap = df["delay_min"].quantile(0.99)
    df["delay_min"] = df["delay_min"].clip(upper=cap)

    return df.reset_index(drop=True)


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Completeness / uniqueness / validity scorecard used as a modelling gate."""
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "completeness_pct": round(100 * df[col].notna().mean(), 2),
                "n_unique": int(df[col].nunique()),
            }
        )
    return pd.DataFrame(rows)


def flag_anomalies(df: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    """Isolation Forest flags on trip duration and cost per stop."""
    from sklearn.ensemble import IsolationForest

    cols = [c for c in ("delay_min", "distance_km", "load_kg") if c in df.columns]
    model = IsolationForest(contamination=contamination, random_state=42)
    df = df.copy()
    df["is_anomaly"] = (model.fit_predict(df[cols].fillna(0)) == -1).astype(int)
    return df


def weekly_sku_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order lines to the SKU-week grain used by the forecaster."""
    out = (
        df.set_index("order_date")
        .groupby([pd.Grouper(freq="W"), "sku", "dc"])["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"order_date": "week"})
    )
    return out.sort_values(["sku", "week"]).reset_index(drop=True)


if __name__ == "__main__":
    print("Run as a module: from data_prep import load_orders")
