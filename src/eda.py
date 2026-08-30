"""Phase 3: exploratory data analysis and visualisation."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_weekly_demand(orders: pd.DataFrame, out: str = "weekly_demand.png") -> str:
    weekly = (
        orders.set_index("order_date")
        .groupby([pd.Grouper(freq="W"), "dc"])["quantity"]
        .sum()
        .reset_index()
    )
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=weekly, x="order_date", y="quantity", hue="dc")
    plt.title("Weekly demand by distribution centre")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_late_rate_by_hour(orders: pd.DataFrame, out: str = "late_by_hour.png") -> str:
    late = orders.groupby("dispatch_hour")["is_late"].mean() * 100
    plt.figure(figsize=(10, 4))
    late.plot(kind="bar", color="#2E75B6")
    plt.ylabel("Late deliveries (%)")
    plt.title("Late-delivery rate by dispatch hour")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_delay_distribution(orders: pd.DataFrame, out: str = "delay_dist.png") -> str:
    plt.figure(figsize=(10, 4))
    sns.histplot(orders["delay_min"], bins=60, kde=True, color="#1F3864")
    plt.axvline(0, color="red", linestyle="--", label="Promised time")
    plt.xlabel("Delay (minutes)")
    plt.title("Distribution of delivery delay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def abc_xyz_classification(weekly_sku: pd.DataFrame) -> pd.DataFrame:
    """ABC by revenue contribution, XYZ by demand variability (CoV)."""
    agg = weekly_sku.groupby("sku")["quantity"].agg(["sum", "mean", "std"])
    agg["cov"] = agg["std"] / agg["mean"].replace(0, pd.NA)

    agg = agg.sort_values("sum", ascending=False)
    agg["cum_share"] = agg["sum"].cumsum() / agg["sum"].sum()
    agg["abc"] = pd.cut(
        agg["cum_share"], bins=[0, 0.8, 0.95, 1.0], labels=["A", "B", "C"]
    )
    agg["xyz"] = pd.cut(
        agg["cov"].fillna(9), bins=[-0.01, 0.5, 1.0, 99], labels=["X", "Y", "Z"]
    )
    agg["segment"] = agg["abc"].astype(str) + agg["xyz"].astype(str)
    return agg.reset_index()
