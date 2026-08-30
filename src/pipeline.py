"""End-to-end daily logistics analytics pipeline.

Mirrors the pseudocode in Section 6.9 of the strategic planning report.

    BEGIN daily_logistics_pipeline
        LOAD and CLEAN orders / inventory / weather
        COMPUTE baseline KPIs
        FORECAST demand per SKU per DC
        RAISE replenishment where stock <= reorder point
        CLUSTER stops INTO capacity-feasible territories
        SOLVE CVRPTW per zone
        SCORE orders for late-delivery risk and re-sequence
        PUBLISH routes and KPIs
    END
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data_prep import data_quality_report, load_orders, weekly_sku_demand
from delay_risk_model import score_orders, train_delay_classifier
from demand_forecasting import make_features, reorder_point, train_forecaster
from kpi_metrics import compute_kpis, kpi_scorecard
from route_optimization import cluster_territories, haversine_matrix, solve_vrp

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("pipeline")

SERVICE_LEVEL = 0.95
LEAD_TIME_DAYS = 5


def run(orders_path: str, trips: pd.DataFrame, inventory: pd.DataFrame, cogs: float):
    log.info("Phase 1-2: load and clean")
    orders = load_orders(orders_path)
    log.info("\n%s", data_quality_report(orders).to_string(index=False))

    log.info("Phase 3: KPI baseline")
    kpis = compute_kpis(orders, trips, inventory, cogs)
    log.info("\n%s", kpi_scorecard(kpis).to_string(index=False))

    log.info("Phase 4: demand forecasting")
    feat = make_features(weekly_sku_demand(orders))
    model, features, metrics = train_forecaster(feat)
    log.info("Forecast metrics: %s", metrics)

    log.info("Phase 5: replenishment policy")
    replenish = []
    for sku, grp in feat.groupby("sku"):
        daily_mean = grp["quantity"].mean() / 7
        daily_std = grp["quantity"].std(ddof=0) / 7
        rop, ss = reorder_point(daily_mean, daily_std, LEAD_TIME_DAYS, SERVICE_LEVEL)
        on_hand = float(inventory.loc[inventory["sku"] == sku, "units"].sum())
        if on_hand <= rop:
            replenish.append(
                {"sku": sku, "on_hand": on_hand, "reorder_point": round(rop, 1),
                 "safety_stock": round(ss, 1)}
            )
    log.info("Replenishment orders raised: %d", len(replenish))

    log.info("Phase 5: territories and routing")
    stops = orders.drop_duplicates("order_id")[["order_id", "lat", "lon", "weight_kg"]]
    stops = cluster_territories(stops)
    plans = {}
    for zone, grp in stops.groupby("zone"):
        matrix = np.round(haversine_matrix(grp["lat"].values, grp["lon"].values) * 1000)
        demands = [0] + grp["weight_kg"].round().astype(int).tolist()[1:]
        try:
            plans[zone] = solve_vrp(matrix, demands, [800, 800])
        except RuntimeError:
            log.warning("Zone %s infeasible with current fleet", zone)

    log.info("Phase 4: delay-risk scoring")
    clf, feats, report = train_delay_classifier(orders)
    log.info("Delay model ROC-AUC: %.3f", report["roc_auc"])
    scored = score_orders(clf, orders, feats)

    return {
        "kpis": kpis,
        "forecast_metrics": metrics,
        "replenishment": pd.DataFrame(replenish),
        "route_plans": plans,
        "at_risk_orders": scored.head(100),
    }


if __name__ == "__main__":
    log.info("Provide real data paths to execute; see README.md for the roadmap.")
