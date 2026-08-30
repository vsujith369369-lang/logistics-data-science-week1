"""Week 3 - automated insight extraction and recommendation ranking.

Reads the simulated shipment dataset, recomputes the diagnostics that back the
report narrative, and emits a ranked action list so the findings can be
refreshed on new data without rewriting prose.

Run:  python src/insights.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eda_analysis import load

OUT = Path(__file__).resolve().parents[1] / "insights.json"

OTD_TARGET = 90.0
FILL_TARGET = 82.0
RETURN_COST_INR = 380.0


def lane_concentration(df: pd.DataFrame, top_n: int = 12) -> dict:
    late = df[df.on_time == 0]
    lanes = (late.groupby(["origin_hub", "carrier"]).size()
             .sort_values(ascending=False))
    share = lanes.head(top_n).sum() / len(late) * 100
    return {
        "late_shipments": int(len(late)),
        "lanes_total": int(lanes.size),
        "top_n": top_n,
        "share_of_failures_pct": round(float(share), 1),
        "worst_lanes": [
            {"hub": h, "carrier": c, "late": int(v)}
            for (h, c), v in lanes.head(top_n).items()
        ],
    }


def carrier_gap(df: pd.DataFrame) -> dict:
    g = df.groupby("carrier").agg(otd=("on_time", "mean"),
                                  cpk=("cost_per_km", "median"))
    g["otd"] *= 100
    best, worst = g.otd.idxmax(), g.otd.idxmin()
    return {
        "best": {"carrier": best, "otd_pct": round(g.otd[best], 2),
                 "cost_per_km": round(g.cpk[best], 2)},
        "worst": {"carrier": worst, "otd_pct": round(g.otd[worst], 2),
                  "cost_per_km": round(g.cpk[worst], 2)},
        "otd_spread_pp": round(g.otd[best] - g.otd[worst], 1),
        "cost_premium_per_km": round(g.cpk[best] - g.cpk[worst], 2),
    }


def service_design_failures(df: pd.DataFrame) -> list[dict]:
    out = []
    for tier, sub in df.groupby("service_level"):
        sla = float(sub.promised_hours.iloc[0])
        mean_transit = float(sub.transit_hours.mean())
        if mean_transit > sla:
            out.append({
                "service_level": tier, "promised_hours": sla,
                "mean_transit_hours": round(mean_transit, 2),
                "otd_pct": round(sub.on_time.mean() * 100, 2),
                "verdict": "structurally undeliverable as specified",
            })
    return sorted(out, key=lambda r: r["otd_pct"])


def seasonal_gap(df: pd.DataFrame) -> dict:
    m = df.set_index("order_date").groupby(pd.Grouper(freq="MS")).agg(
        otd=("on_time", "mean"), vol=("shipment_id", "count"),
        cost=("freight_cost_inr", "mean"))
    m["otd"] *= 100
    best, worst = m.otd.idxmax(), m.otd.idxmin()
    return {
        "best_month": str(best.date()), "best_otd_pct": round(m.otd[best], 2),
        "worst_month": str(worst.date()), "worst_otd_pct": round(m.otd[worst], 2),
        "otd_swing_pp": round(m.otd[best] - m.otd[worst], 1),
        "volume_swing_pct": round((m.vol.max() / m.vol.min() - 1) * 100, 1),
        "peak_cost_surcharge_pct": round((m.cost[worst] / m.cost[best] - 1) * 100, 1),
    }


def cost_of_unreliability(df: pd.DataFrame) -> dict:
    late, ontime = df[df.on_time == 0], df[df.on_time == 1]
    excess = late.returned.mean() - ontime.returned.mean()
    extra_returns = excess * len(late)
    return {
        "excess_return_rate_pp": round(excess * 100, 2),
        "avoidable_returns": int(round(extra_returns)),
        "direct_cost_inr_lakh": round(extra_returns * RETURN_COST_INR / 1e5, 2),
        "csat_penalty": round(ontime.csat_score.mean() - late.csat_score.mean(), 2),
    }


def scale_economies(df: pd.DataFrame) -> dict:
    d = df.assign(dec=pd.qcut(df.distance_km, 10, labels=False) + 1)
    cpk = d.groupby("dec").cost_per_km.median()
    band = pd.cut(df.vehicle_fill_rate, [0, .5, .65, .8, .9, 1.0])
    per_kg = df.groupby(band, observed=True).cost_per_kg.median()
    return {
        "cpk_shortest_decile": round(float(cpk.iloc[0]), 2),
        "cpk_longest_decile": round(float(cpk.iloc[-1]), 2),
        "unit_cost_reduction_pct": round((1 - cpk.iloc[-1] / cpk.iloc[0]) * 100, 1),
        "cost_per_kg_spread_pct": round(
            (per_kg.max() / per_kg.min() - 1) * 100, 1),
        "fill_rate_pct": round(df.vehicle_fill_rate.mean() * 100, 2),
        "fill_gap_pp": round(FILL_TARGET - df.vehicle_fill_rate.mean() * 100, 1),
    }


def rank_recommendations(f: dict) -> list[dict]:
    """Score actions by impact x confidence / effort, then order them."""
    recs = [
        {"action": "Re-route the top 12 failing hub-carrier lanes to the "
                   "three best-performing carriers",
         "evidence": f"{f['lanes']['share_of_failures_pct']}% of "
                     f"{f['lanes']['late_shipments']} late shipments",
         "impact": 9, "confidence": 9, "effort": 3, "horizon": "30 days"},
        {"action": "Re-specify Same-Day as a distance-capped 18-hour product",
         "evidence": "mean transit exceeds the contractual SLA",
         "impact": 8, "confidence": 9, "effort": 4, "horizon": "45 days"},
        {"action": "Pre-book Oct-Nov peak capacity at fixed rates in Q2",
         "evidence": f"{f['season']['otd_swing_pp']} pp seasonal OTD swing on "
                     f"{f['season']['volume_swing_pct']}% volume swing",
         "impact": 8, "confidence": 7, "effort": 5, "horizon": "1 quarter"},
        {"action": "Delhi hub capacity programme (sortation shift + dispatch windows)",
         "evidence": "largest absolute failure pool of any origin hub",
         "impact": 7, "confidence": 7, "effort": 6, "horizon": "1 quarter"},
        {"action": "Multi-drop consolidation of sub-350 km loads plus load-level costing",
         "evidence": f"unit cost falls {f['scale']['unit_cost_reduction_pct']}% "
                     f"across haul deciles while fill-rate bands differ by only "
                     f"{f['scale']['cost_per_kg_spread_pct']}%",
         "impact": 6, "confidence": 6, "effort": 6, "horizon": "60 days"},
        {"action": "Performance-linked carrier scorecards with volume reallocation",
         "evidence": f"{f['carrier']['otd_spread_pp']} pp spread between best "
                     f"and worst carrier",
         "impact": 6, "confidence": 8, "effort": 4, "horizon": "ongoing"},
    ]
    for r in recs:
        r["score"] = round(r["impact"] * r["confidence"] / r["effort"], 2)
    recs.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(recs, 1):
        r["rank"] = i
    return recs


def main() -> None:
    df = load()
    findings = {
        "kpi": {
            "shipments": int(len(df)),
            "otd_pct": round(df.on_time.mean() * 100, 2),
            "otd_gap_pp": round(OTD_TARGET - df.on_time.mean() * 100, 2),
            "avg_cost_inr": round(df.freight_cost_inr.mean(), 2),
        },
        "lanes": lane_concentration(df),
        "carrier": carrier_gap(df),
        "service_design": service_design_failures(df),
        "season": seasonal_gap(df),
        "unreliability_cost": cost_of_unreliability(df),
        "scale": scale_economies(df),
    }
    findings["recommendations"] = rank_recommendations(findings)
    OUT.write_text(json.dumps(findings, indent=2))
    print(f"wrote {OUT}")
    for r in findings["recommendations"]:
        print(f"  {r['rank']}. [{r['score']:>5}] {r['action']}")


if __name__ == "__main__":
    main()
