"""Week 3 - Hypothetical logistics dataset simulation.

Generates a reproducible 60,000-row shipment-level dataset for NorthStar
Logistics covering 24 months of operations across 5 regional hubs,
6 carriers, 4 service levels and 5 product categories.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_ROWS = 60_000
START, END = "2024-01-01", "2025-12-31"

HUBS = {
    "Mumbai":    {"share": 0.28, "cost": 1.00, "delay": 1.00},
    "Delhi":     {"share": 0.24, "cost": 1.05, "delay": 1.22},
    "Bengaluru": {"share": 0.20, "cost": 0.96, "delay": 0.88},
    "Chennai":   {"share": 0.16, "cost": 0.94, "delay": 0.95},
    "Kolkata":   {"share": 0.12, "cost": 1.02, "delay": 1.35},
}
CARRIERS = {
    "BlueDart":  {"share": 0.22, "cost": 1.18, "delay": 0.78},
    "Delhivery": {"share": 0.21, "cost": 1.00, "delay": 0.95},
    "Ecom":      {"share": 0.17, "cost": 0.92, "delay": 1.12},
    "XpressBees":{"share": 0.15, "cost": 0.90, "delay": 1.20},
    "SafeExp":   {"share": 0.13, "cost": 1.08, "delay": 0.88},
    "LocalFleet":{"share": 0.12, "cost": 0.84, "delay": 1.45},
}
SERVICE = {
    "Same-Day": {"share": 0.08, "sla": 12,  "cost": 2.40},
    "Express":  {"share": 0.27, "sla": 36,  "cost": 1.55},
    "Standard": {"share": 0.49, "sla": 72,  "cost": 1.00},
    "Economy":  {"share": 0.16, "sla": 120, "cost": 0.72},
}
CATEGORY = {
    "Electronics": {"share": 0.22, "wt": 2.4,  "val": 14500},
    "Apparel":     {"share": 0.30, "wt": 0.9,  "val": 1900},
    "Grocery":     {"share": 0.21, "wt": 5.2,  "val": 1200},
    "Furniture":   {"share": 0.09, "wt": 24.0, "val": 11800},
    "Pharma":      {"share": 0.18, "wt": 0.6,  "val": 2400},
}


def _pick(spec: dict) -> np.ndarray:
    keys = list(spec)
    probs = np.array([spec[k]["share"] for k in keys], dtype=float)
    return RNG.choice(keys, size=N_ROWS, p=probs / probs.sum())


def simulate() -> pd.DataFrame:
    dates = pd.to_datetime(RNG.integers(0, (pd.Timestamp(END) - pd.Timestamp(START)).days + 1, N_ROWS),
                           unit="D", origin=pd.Timestamp(START))
    hub, carrier, service, category = _pick(HUBS), _pick(CARRIERS), _pick(SERVICE), _pick(CATEGORY)

    df = pd.DataFrame({
        "shipment_id": [f"NS{100000 + i}" for i in range(N_ROWS)],
        "order_date": dates,
        "origin_hub": hub,
        "carrier": carrier,
        "service_level": service,
        "product_category": category,
    })

    # ---- physical attributes -------------------------------------------------
    base_wt = np.array([CATEGORY[c]["wt"] for c in category])
    df["weight_kg"] = np.round(RNG.gamma(2.0, base_wt / 2.0).clip(0.1, 400), 2)
    df["units"] = RNG.integers(1, 12, N_ROWS)
    df["distance_km"] = np.round(RNG.lognormal(5.9, 0.75, N_ROWS).clip(8, 2600), 1)
    df["order_value_inr"] = np.round(
        np.array([CATEGORY[c]["val"] for c in category]) * RNG.lognormal(0, 0.45, N_ROWS), 2)

    # ---- seasonality: festive Oct-Nov peak + weekend dip ---------------------
    month, dow = df.order_date.dt.month.values, df.order_date.dt.dayofweek.values
    season = 1 + 0.42 * np.isin(month, [10, 11]) + 0.16 * np.isin(month, [3, 7]) - 0.10 * (dow >= 5)

    # ---- transit time --------------------------------------------------------
    sla = np.array([SERVICE[s]["sla"] for s in service])
    base_hours = 0.55 * sla + 0.021 * df.distance_km.values
    stress = (np.array([HUBS[h]["delay"] for h in hub])
              * np.array([CARRIERS[c]["delay"] for c in carrier])
              * season)
    df["transit_hours"] = np.round((base_hours * stress * RNG.lognormal(0, 0.28, N_ROWS)).clip(2, 460), 2)
    df["promised_hours"] = sla
    df["on_time"] = (df.transit_hours <= df.promised_hours).astype(int)

    # ---- cost ----------------------------------------------------------------
    cost = (48
            + 3.05 * df.distance_km.values ** 0.72
            + 11.5 * df.weight_kg.values ** 0.85
            + 0.004 * df.order_value_inr.values)
    cost *= (np.array([SERVICE[s]["cost"] for s in service])
             * np.array([CARRIERS[c]["cost"] for c in carrier])
             * np.array([HUBS[h]["cost"] for h in hub])
             * (1 + 0.11 * np.isin(month, [10, 11])))
    df["freight_cost_inr"] = np.round(cost * RNG.lognormal(0, 0.16, N_ROWS), 2)

    # ---- downstream outcomes -------------------------------------------------
    late = 1 - df.on_time.values
    p_ret = 0.028 + 0.061 * late + 0.030 * (category == "Apparel") + 0.018 * (category == "Electronics")
    df["returned"] = RNG.binomial(1, p_ret.clip(0, 0.6))
    df["vehicle_fill_rate"] = np.round(RNG.beta(6, 2.4, N_ROWS).clip(0.15, 0.99), 3)
    df["csat_score"] = np.round((5.0 - 1.55 * late - 0.9 * df.returned
                                 + RNG.normal(0, 0.42, N_ROWS)).clip(1, 5), 2)
    return df.sort_values("order_date").reset_index(drop=True)


if __name__ == "__main__":
    d = simulate()
    d.to_csv("/tmp/wk3/logistics_shipments.csv", index=False)
    print(d.shape)
