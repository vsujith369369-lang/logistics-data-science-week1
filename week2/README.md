# Week 2 — Data Collection and Preparation in Logistics

Documented preprocessing pipeline for the NorthStar Logistics case study.

**Report:** [`docs/Week2_Data_Collection_and_Preparation_Report.docx`](../docs/Week2_Data_Collection_and_Preparation_Report.docx)

## Pipeline stages

| Stage | Module | Purpose |
|---|---|---|
| 0 | `src/ingest.py` | Load WMS CSV, TMS JSON and the Excel cost ledger |
| 0 | `src/profiling.py` | Quantify defects before repairing anything |
| 1–4 | `src/cleaning.py` | Schema enforcement, deduplication, text standardisation, business rules |
| 5 | `src/imputation.py` | Column-wise missing-value strategies (MCAR / MAR / MNAR aware) |
| 6 | `src/outliers.py` | Domain limits, IQR, modified Z-score (MAD), Isolation Forest |
| 7 | `src/features.py` | Derived logistics features (transit hours, OTD, cost/km, utilisation) |
| 8 | `src/scaling.py` | Log1p, robust/standard/min-max scaling, one-hot and cyclical encoding |
| 9 | `src/validation.py` | 16 automated assertions and the quality scorecard |
| — | `src/preprocessing_pipeline.py` | End-to-end orchestration |

## Results

- 118,000 raw rows → 112,400 analysis-ready rows (95.3% retained)
- 0 nulls in modelling columns, 0 duplicate order IDs
- 31 carrier spelling variants → 9 canonical values
- freight_cost skew 3.42 → 0.31 after log1p
- Measured on-time delivery: 91.4% (biased, raw) → 88.1% (true, cleaned)

## Run

```bash
pip install -r ../requirements.txt
python src/preprocessing_pipeline.py
```
