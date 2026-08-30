"""Phase 3: KPI computation for the five logistics KPIs defined in the report."""

from __future__ import annotations

import pandas as pd

TARGETS = {
    "on_time_delivery_pct": 95.0,
    "cost_per_delivery": 53.0,       # lower is better
    "inventory_turns": 8.0,
    "order_fill_rate_pct": 98.0,
    "capacity_utilisation_pct": 85.0,
}


def compute_kpis(
    orders: pd.DataFrame,
    trips: pd.DataFrame,
    inventory: pd.DataFrame,
    cogs: float,
) -> dict:
    """Return the five headline KPIs."""
    otd = 100 * (1 - orders["is_late"].mean())
    cpd = trips["total_cost"].sum() / orders["order_id"].nunique()
    turns = cogs / inventory["value"].mean()
    fill = 100 * (orders["qty_shipped"] >= orders["quantity"]).mean()
    util = 100 * (trips["load_kg"] / trips["capacity_kg"]).mean()

    return {
        "on_time_delivery_pct": round(otd, 2),
        "cost_per_delivery": round(cpd, 2),
        "inventory_turns": round(turns, 2),
        "order_fill_rate_pct": round(fill, 2),
        "capacity_utilisation_pct": round(util, 2),
    }


def kpi_scorecard(kpis: dict) -> pd.DataFrame:
    """Compare actuals against targets; cost is the only lower-is-better KPI."""
    rows = []
    for name, value in kpis.items():
        target = TARGETS[name]
        met = value <= target if name == "cost_per_delivery" else value >= target
        rows.append(
            {"kpi": name, "actual": value, "target": target, "target_met": bool(met)}
        )
    return pd.DataFrame(rows)


def kpi_by_dimension(orders: pd.DataFrame, dimension: str = "dc") -> pd.DataFrame:
    """Break on-time delivery and delay down by DC, zone or month."""
    return (
        orders.groupby(dimension)
        .agg(
            orders=("order_id", "nunique"),
            on_time_pct=("is_late", lambda s: round(100 * (1 - s.mean()), 2)),
            avg_delay_min=("delay_min", "mean"),
        )
        .reset_index()
    )
