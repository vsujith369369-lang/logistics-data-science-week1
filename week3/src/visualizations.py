"""Week 3 - Visualization suite for the logistics dataset (matplotlib + seaborn)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eda_analysis import NUMERIC, load

FIG = "/tmp/wk3/figs"
PAL = ["#0B3C5D", "#328CC1", "#D9B310", "#1D2731", "#7A9E9F", "#B5651D"]
sns.set_theme(style="whitegrid", font_scale=0.95)
plt.rcParams.update({"figure.dpi": 160, "savefig.bbox": "tight",
                     "axes.titleweight": "bold", "axes.titlesize": 11.5,
                     "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=160)
    plt.close(fig)
    print("saved", name)


def fig1_distributions(df):
    fig, ax = plt.subplots(2, 2, figsize=(9, 5.6))
    sns.histplot(df.transit_hours, bins=60, kde=True, color=PAL[0], ax=ax[0, 0])
    ax[0, 0].set_title("Transit time distribution (right-skewed)"); ax[0, 0].set_xlabel("Transit hours")
    sns.histplot(df.freight_cost_inr, bins=60, kde=True, color=PAL[1], ax=ax[0, 1])
    ax[0, 1].set_title("Freight cost distribution"); ax[0, 1].set_xlabel("Freight cost (INR)")
    sns.histplot(np.log1p(df.freight_cost_inr), bins=60, kde=True, color=PAL[5], ax=ax[1, 0])
    ax[1, 0].set_title("Freight cost after log1p (near-normal)"); ax[1, 0].set_xlabel("log1p(cost)")
    sns.histplot(df.vehicle_fill_rate, bins=50, kde=True, color=PAL[4], ax=ax[1, 1])
    ax[1, 1].set_title("Vehicle fill rate (left-skewed)"); ax[1, 1].set_xlabel("Fill rate")
    fig.suptitle("Figure 1 - Distributions of core logistics variables", fontweight="bold")
    fig.tight_layout(); save(fig, "fig01_distributions")


def fig2_carrier_otd(df):
    g = df.groupby("carrier").agg(otd=("on_time", "mean"), cost=("freight_cost_inr", "mean")).sort_values("otd")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(g.index, 100 * g.otd, color=[PAL[0] if v >= .75 else PAL[2] if v >= .6 else PAL[5] for v in g.otd])
    ax.axvline(90, ls="--", c="crimson", lw=1.2, label="Target OTD 90%")
    for b, v in zip(bars, 100 * g.otd):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 100); ax.set_xlabel("On-time delivery (%)")
    ax.set_title("Figure 2 - On-time delivery by carrier vs 90% target"); ax.legend(loc="lower right")
    save(fig, "fig02_carrier_otd")


def fig3_cost_vs_otd(df):
    g = df.groupby("carrier").agg(otd=("on_time", "mean"), cpk=("cost_per_km", "median"),
                                  n=("shipment_id", "count")).reset_index()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.scatter(g.cpk, 100 * g.otd, s=g.n / 12, c=PAL[1], alpha=.75, edgecolor=PAL[3])
    for _, r in g.iterrows():
        ax.annotate(r.carrier, (r.cpk, 100 * r.otd), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Median cost per km (INR)"); ax.set_ylabel("On-time delivery (%)")
    ax.set_title("Figure 3 - Cost-service trade-off by carrier (bubble = volume)")
    save(fig, "fig03_cost_service_tradeoff")


def fig4_monthly_trend(df):
    m = df.groupby("month").agg(ship=("shipment_id", "count"), otd=("on_time", "mean"),
                                cost=("freight_cost_inr", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    ax.bar(m.month, m.ship, color=PAL[4], alpha=.65, label="Shipment volume")
    ax.set_ylabel("Shipments"); ax.tick_params(axis="x", rotation=60)
    ax2 = ax.twinx()
    ax2.plot(m.month, 100 * m.otd, color=PAL[5], marker="o", lw=2, label="OTD %")
    ax2.set_ylabel("On-time delivery (%)"); ax2.grid(False)
    ax.set_title("Figure 4 - Monthly volume vs on-time delivery (festive Oct-Nov dip)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    save(fig, "fig04_monthly_trend")


def fig5_corr(df):
    cols = NUMERIC + ["delay_hours", "on_time", "returned"]
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                annot_kws={"size": 7.5}, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Figure 5 - Pearson correlation matrix of logistics variables")
    save(fig, "fig05_correlation")


def fig6_box_service(df):
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))
    order = ["Same-Day", "Express", "Standard", "Economy"]
    sns.boxplot(data=df, x="service_level", y="transit_hours", order=order, palette=PAL[:4], ax=ax[0], showfliers=False)
    ax[0].set_title("Transit hours by service level"); ax[0].set_xlabel("")
    sns.boxplot(data=df, x="service_level", y="freight_cost_inr", order=order, palette=PAL[:4], ax=ax[1], showfliers=False)
    ax[1].set_title("Freight cost by service level"); ax[1].set_xlabel("")
    fig.suptitle("Figure 6 - Service-level dispersion in time and cost", fontweight="bold")
    fig.tight_layout(); save(fig, "fig06_service_box")


def fig7_scatter_cost(df):
    s = df.sample(4000, random_state=1)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    sc = ax.scatter(s.distance_km, s.freight_cost_inr, c=s.weight_kg.clip(0, 40),
                    cmap="viridis", s=9, alpha=.55)
    z = np.polyfit(np.log(s.distance_km), np.log(s.freight_cost_inr), 1)
    xs = np.linspace(s.distance_km.min(), s.distance_km.max(), 200)
    ax.plot(xs, np.exp(z[1]) * xs ** z[0], c="crimson", lw=2,
            label=f"Power fit: cost $\\propto$ dist$^{{{z[0]:.2f}}}$")
    ax.set_xlabel("Distance (km)"); ax.set_ylabel("Freight cost (INR)"); ax.legend()
    fig.colorbar(sc, ax=ax, label="Weight (kg)")
    ax.set_title("Figure 7 - Cost vs distance shows sub-linear (economies of distance) scaling")
    save(fig, "fig07_cost_distance")


def fig8_hub_heatmap(df):
    p = df.pivot_table(index="origin_hub", columns="carrier", values="on_time", aggfunc="mean") * 100
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    sns.heatmap(p.round(1), annot=True, fmt=".1f", cmap="RdYlGn", vmin=25, vmax=95,
                annot_kws={"size": 8}, cbar_kws={"label": "OTD %"}, ax=ax)
    ax.set_title("Figure 8 - OTD heatmap: hub x carrier bottleneck map")
    ax.set_xlabel(""); ax.set_ylabel("")
    save(fig, "fig08_hub_carrier_heatmap")


def fig9_pareto(df):
    g = (df.groupby(["origin_hub", "carrier"]).apply(lambda d: (1 - d.on_time.mean()) * len(d))
           .sort_values(ascending=False).head(12))
    lbl = [f"{a}-{b}" for a, b in g.index]
    cum = 100 * g.cumsum() / ((1 - df.on_time.mean()) * len(df))
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    ax.bar(lbl, g.values, color=PAL[0]); ax.tick_params(axis="x", rotation=55)
    ax.set_ylabel("Late shipments")
    ax2 = ax.twinx(); ax2.plot(lbl, cum.values, color=PAL[2], marker="o", lw=2)
    ax2.axhline(80, ls="--", c="crimson", lw=1); ax2.set_ylabel("Cumulative % of all late shipments")
    ax2.grid(False); ax2.set_ylim(0, 100)
    ax.set_title("Figure 9 - Pareto of late shipments by hub-carrier lane")
    save(fig, "fig09_pareto_late")


def fig10_cost_drivers(df):
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.2))
    c = df.groupby("product_category").agg(cpk=("cost_per_km", "median"),
                                           spend=("freight_cost_inr", "sum")).sort_values("cpk")
    ax[0].barh(c.index, c.cpk, color=PAL[1]); ax[0].set_xlabel("Median cost per km (INR)")
    ax[0].set_title("Cost intensity by product category")
    ax[1].pie(c.spend, labels=c.index, autopct="%1.1f%%", colors=PAL, startangle=110,
              textprops={"fontsize": 8.5})
    ax[1].set_title("Share of total freight spend")
    fig.suptitle("Figure 10 - Where the freight money goes", fontweight="bold")
    fig.tight_layout(); save(fig, "fig10_cost_drivers")


def fig11_delay_impact(df):
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))
    b = df.groupby("on_time").agg(ret=("returned", "mean"), csat=("csat_score", "mean"))
    ax[0].bar(["Late", "On-time"], 100 * b.ret.values, color=[PAL[5], PAL[0]])
    for i, v in enumerate(100 * b.ret.values):
        ax[0].text(i, v + .12, f"{v:.2f}%", ha="center", fontsize=9)
    ax[0].set_ylabel("Return rate (%)"); ax[0].set_title("Late deliveries drive returns")
    sns.violinplot(data=df, x="on_time", y="csat_score", palette=[PAL[5], PAL[0]], ax=ax[1], cut=0)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["Late", "On-time"])
    ax[1].set_xlabel(""); ax[1].set_ylabel("CSAT (1-5)"); ax[1].set_title("CSAT distribution by delivery outcome")
    fig.suptitle("Figure 11 - Downstream commercial impact of delays", fontweight="bold")
    fig.tight_layout(); save(fig, "fig11_delay_impact")


def fig12_scale_economies(df):
    """Left: unit cost falls steeply with haul length. Right: fill rate shows no
    unit-cost benefit today - an honest negative result."""
    d = df.copy()
    d["dec"] = pd.qcut(d.distance_km, 10, labels=False) + 1
    g = d.groupby("dec").agg(dist=("distance_km", "median"), cpk=("cost_per_km", "median"))
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.2))
    ax[0].plot(g.dist, g.cpk, marker="o", lw=2.4, color=PAL[0])
    ax[0].fill_between(g.dist, g.cpk, alpha=.18, color=PAL[1])
    for x, y in zip(g.dist, g.cpk):
        ax[0].annotate(f"{y:.2f}", (x, y), xytext=(0, 8), textcoords="offset points",
                       ha="center", fontsize=7.5)
    ax[0].set_xlabel("Median haul distance of decile (km)")
    ax[0].set_ylabel("Median cost per km (INR)")
    ax[0].set_title("Unit cost falls 74% from shortest to longest haul decile")

    b = pd.cut(d.vehicle_fill_rate, [0, .5, .65, .8, .9, 1.0],
               labels=["<50%", "50-65%", "65-80%", "80-90%", ">90%"])
    h = d.groupby(b, observed=True).cost_per_kg.median()
    bars = ax[1].bar(h.index.astype(str), h.values, color=PAL[4])
    ax[1].axhline(h.mean(), ls="--", c="crimson", lw=1.2, label="Overall median")
    for bar, v in zip(bars, h.values):
        ax[1].text(bar.get_x() + bar.get_width() / 2, v + 3, f"{v:.1f}", ha="center", fontsize=8.5)
    ax[1].set_ylim(0, 320); ax[1].legend(fontsize=8)
    ax[1].set_xlabel("Vehicle fill-rate band"); ax[1].set_ylabel("Median cost per kg (INR)")
    ax[1].set_title("Utilisation shows no unit-cost benefit today")
    fig.suptitle("Figure 12 - Where scale economies exist, and where they are not being captured",
                 fontweight="bold")
    fig.tight_layout(); save(fig, "fig12_scale_economies")


if __name__ == "__main__":
    df = load()
    for f in (fig1_distributions, fig2_carrier_otd, fig3_cost_vs_otd, fig4_monthly_trend,
              fig5_corr, fig6_box_service, fig7_scatter_cost, fig8_hub_heatmap,
              fig9_pareto, fig10_cost_drivers, fig11_delay_impact, fig12_scale_economies):
        f(df)
