# Week 3 - Exploratory Data Analysis and Visualization in Logistics

Analysis of a simulated 60,000-shipment, 24-month logistics network (NorthStar Logistics)
covering five origin hubs, six carriers, four service levels and five product categories.

Report: [`docs/Week3_Logistics_EDA_and_Visualization_Report.docx`](../docs/Week3_Logistics_EDA_and_Visualization_Report.docx)

## Reproduce

```bash
pip install -r requirements.txt
python src/simulate_data.py      # writes logistics_shipments.csv (60,000 x 17)
python src/eda_analysis.py       # writes eda_stats.json
python src/visualizations.py     # writes figs/fig01..fig12 at 160 dpi
python src/insights.py           # writes insights.json + ranked recommendations
```

All randomness is seeded (`np.random.default_rng(42)`), so results are byte-reproducible.

## Headline KPI baseline

| KPI | Value | Target | Status |
|---|---|---|---|
| On-time delivery | 67.70% | 90.0% | Critical gap |
| Average transit time | 55.63 h | 48.0 h | Behind |
| Average freight cost / shipment | INR 441.53 | INR 410 | Over budget |
| Blended cost per km | INR 0.921 | INR 0.85 | Over budget |
| Return rate | 5.82% | < 4.0% | Behind |
| Vehicle fill rate | 71.43% | 82.0% | Behind |
| Average CSAT | 4.34 / 5 | 4.50 | Slightly behind |

## Key findings

1. Failure is concentrated: 12 of 30 hub-carrier lanes generate 61.6% of all 19,383 late shipments.
2. Carrier choice spans 49.9 points of OTD (BlueDart 85.8% vs LocalFleet 35.9%) for INR 0.33 per km.
3. Same-Day is structurally undeliverable: mean transit 20.45 h against a 12 h promise; OTD 28.2%.
4. Oct-Nov is a capacity-planning failure, not a demand shock - volume moves under 7%, OTD moves 31.8 points.
5. Unit cost falls 74% from the shortest to the longest haul decile; fill-rate bands differ by only 4%,
   exposing shipment-level rather than load-level costing.
6. Unreliability costs about INR 4.21 lakh in avoidable returns and 1.44 points of CSAT.

## Figure index

| File | Chart | Question answered |
|---|---|---|
| fig01_distributions.png | Histograms + KDE | Shape of transit, cost, fill rate |
| fig02_carrier_otd.png | Horizontal bars | Which carriers miss the 90% target |
| fig03_cost_service_tradeoff.png | Bubble scatter | Cost/service efficient frontier |
| fig04_monthly_trend.png | Dual-axis time series | Seasonality of volume vs OTD |
| fig05_correlation.png | Heatmap | Variable interdependence |
| fig06_service_box.png | Box plots | Dispersion inside each service tier |
| fig07_cost_distance.png | Scatter + fit | Cost function shape |
| fig08_hub_carrier_heatmap.png | Matrix heatmap | Lane-level bottleneck map |
| fig09_pareto_late.png | Pareto | Concentration of failures |
| fig10_cost_drivers.png | Ranked bars | What actually drives cost |
| fig11_delay_impact.png | Bar + violin | Commercial impact of lateness |
| fig12_scale_economies.png | Decile curve + bands | Where scale economies exist |

## Layout

```
week3/
  src/simulate_data.py     reproducible dataset generator
  src/eda_analysis.py      descriptive statistics, segment benchmarks, correlations
  src/visualizations.py    all twelve figures (Matplotlib + seaborn)
  src/insights.py          automated insight extraction and recommendation ranking
  figs/                    exported PNG figures at 160 dpi
  requirements.txt
```
