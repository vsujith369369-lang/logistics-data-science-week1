# Week 1 Task: Strategic Planning and Data Exploration in Logistics

Strategic planning deliverable for a data science project in logistics and supply chain analytics.

**Author:** Sujith V

## Deliverables

| File | Description |
|------|-------------|
| `docs/Week1_Logistics_Strategic_Planning_Report.docx` | Full strategic planning report (DOC deliverable) |
| `src/data_prep.py` | Data loading, cleaning and validation |
| `src/kpi_metrics.py` | Computation of the five logistics KPIs |
| `src/eda.py` | Exploratory data analysis and visualisation |
| `src/demand_forecasting.py` | SKU-level demand forecasting + reorder point / EOQ |
| `src/delay_risk_model.py` | Late-delivery risk classifier |
| `src/route_optimization.py` | Territory clustering + CVRP with OR-Tools |
| `src/pipeline.py` | End-to-end pipeline orchestration (pseudocode-style) |
| `requirements.txt` | Python dependencies |

## Scenario

NorthStar Logistics, a mid-sized 3PL, runs three distribution centres, ~4,000 SKUs and ~12,000 last-mile
deliveries per week. It faces stock-outs alongside excess slow-moving inventory, rising cost per delivery
from manually built routes, and on-time delivery that has slipped to 88% against a 95% SLA.

## KPIs

1. **On-Time Delivery Rate (OTD)** - 88% → target ≥ 95%
2. **Cost per Delivery (CPD)** - INR 62 → target ≤ INR 53
3. **Inventory Turnover Ratio** - 6.1x → target ≥ 8.0x
4. **Order Fill Rate** - 93% → target ≥ 98%
5. **Vehicle Capacity Utilisation** - 71% → target ≥ 85%

## Methods

- **Regression / time-series:** SARIMA, Prophet, LightGBM for SKU demand forecasting
- **Classification:** gradient boosting to predict late-delivery risk before dispatch
- **Clustering:** K-Means / DBSCAN for delivery territories, ABC/XYZ for SKU segmentation
- **Optimisation:** CVRPTW with Google OR-Tools; EOQ and safety-stock models

## Roadmap

| Week | Phase |
|------|-------|
| 1-2 | Data collection and integration |
| 2 | Cleaning and validation |
| 3 | Exploratory data analysis and KPI baseline |
| 4-5 | Predictive modelling |
| 6 | Optimisation and prescriptive layer |
| 7-8 | Deployment, reporting and monitoring |

## Public data sources

Olist Brazilian E-Commerce (Kaggle) · UCI Online Retail II · Instacart Market Basket ·
OpenStreetMap / OSRM · CVRPLIB & Solomon VRPTW benchmarks · Open-Meteo historical weather API

## Usage

```bash
pip install -r requirements.txt
python src/pipeline.py
```
