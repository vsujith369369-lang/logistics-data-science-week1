"""Week 3 - Exploratory Data Analysis on the simulated logistics dataset."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

NUMERIC = ["transit_hours", "freight_cost_inr", "distance_km", "weight_kg",
           "order_value_inr", "vehicle_fill_rate", "csat_score"]


def load(path: str = "/tmp/wk3/logistics_shipments.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["order_date"])
    df["month"] = df.order_date.dt.to_period("M").astype(str)
    df["cost_per_km"] = df.freight_cost_inr / df.distance_km
    df["cost_per_kg"] = df.freight_cost_inr / df.weight_kg
    df["delay_hours"] = (df.transit_hours - df.promised_hours).clip(lower=0)
    return df


def central_tendency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in NUMERIC:
        s = df[c]
        rows.append({
            "variable": c, "mean": s.mean(), "median": s.median(),
            "std": s.std(), "skew": s.skew(), "kurtosis": s.kurtosis(),
            "p05": s.quantile(0.05), "p95": s.quantile(0.95), "cv": s.std() / s.mean(),
        })
    return pd.DataFrame(rows).round(3)


def kpi_summary(df: pd.DataFrame) -> dict:
    return {
        "shipments": int(len(df)),
        "otd_pct": round(100 * df.on_time.mean(), 2),
        "avg_transit_h": round(df.transit_hours.mean(), 2),
        "median_transit_h": round(df.transit_hours.median(), 2),
        "avg_cost": round(df.freight_cost_inr.mean(), 2),
        "cost_per_km": round(df.freight_cost_inr.sum() / df.distance_km.sum(), 3),
        "return_pct": round(100 * df.returned.mean(), 2),
        "fill_rate_pct": round(100 * df.vehicle_fill_rate.mean(), 2),
        "csat": round(df.csat_score.mean(), 2),
        "total_spend_cr": round(df.freight_cost_inr.sum() / 1e7, 2),
        "avg_delay_late_h": round(df.loc[df.on_time == 0, "delay_hours"].mean(), 2),
    }


def group_table(df: pd.DataFrame, key: str) -> pd.DataFrame:
    g = df.groupby(key).agg(
        shipments=("shipment_id", "count"),
        otd_pct=("on_time", lambda s: 100 * s.mean()),
        avg_transit_h=("transit_hours", "mean"),
        avg_cost=("freight_cost_inr", "mean"),
        cost_per_km=("cost_per_km", "median"),
        return_pct=("returned", lambda s: 100 * s.mean()),
        csat=("csat_score", "mean"),
    ).round(2).sort_values("otd_pct", ascending=False)
    return g.reset_index()


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = NUMERIC + ["on_time", "returned", "delay_hours"]
    return df[cols].corr(method="pearson").round(3)


if __name__ == "__main__":
    df = load()
    out = {
        "kpi": kpi_summary(df),
        "central": central_tendency(df).to_dict("records"),
        "carrier": group_table(df, "carrier").to_dict("records"),
        "hub": group_table(df, "origin_hub").to_dict("records"),
        "service": group_table(df, "service_level").to_dict("records"),
        "category": group_table(df, "product_category").to_dict("records"),
        "corr": correlations(df).to_dict(),
        "monthly": df.groupby("month").agg(
            shipments=("shipment_id", "count"),
            otd=("on_time", lambda s: 100 * s.mean()),
            cost=("freight_cost_inr", "mean")).round(2).reset_index().to_dict("records"),
    }
    json.dump(out, open("/tmp/wk3/eda_stats.json", "w"), indent=1, default=float)
    print(json.dumps(out["kpi"], indent=1))
    print(pd.DataFrame(out["carrier"]).to_string(index=False))
    print(pd.DataFrame(out["hub"]).to_string(index=False))
    print(pd.DataFrame(out["service"]).to_string(index=False))
    print(pd.DataFrame(out["category"]).to_string(index=False))
    print(central_tendency(df).to_string(index=False))
