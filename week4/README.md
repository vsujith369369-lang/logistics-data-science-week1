# Week 4 — Predictive Modelling and Optimisation in Logistics

Forecasting shipment transit time for NorthStar Logistics and converting the predictions into cost-optimal operating decisions.

**Report:** `Week4_Predictive_Modeling_and_Optimization_Report.docx` (22 pages, 13 figures)

## Headline results
- Dataset: 60,000 simulated shipments, 25 attributes, seed 42
- Champion model: tuned HistGradientBoostingRegressor — MAE 3.05 h, RMSE 3.88 h, R² 0.9563, MAPE 9.13%
- Baselines: Ridge (MAE 4.29), Decision Tree (MAE 6.49), Random Forest (MAE 4.79)
- Late-risk classifier: ROC-AUC 0.9962, PR-AUC 0.9763 (base late rate 13.11%)
- Risk-triggered expediting at p(late) ≥ 0.40: 13.39% of volume, 92.3% recall, 90.4% precision, INR 4.21 lakh net saving on the hold-out book
- Carrier re-allocation under ±25% capacity caps: 6.28% lower expected landed cost
- Consolidation of partial loads: 45.4% of shipments qualify, INR 7.05 lakh annualised

## Method
Leak-safe ColumnTransformer pipeline (imputation, scaling, one-hot), four benchmarked model families, 3-fold cross-validation, randomised hyper-parameter search over a 72-point grid, residual diagnostics, permutation importance and segment-level fairness checks, followed by expected-cost optimisation policies, a monitoring plan and six time-bound recommendations.

## Stack
Python · pandas · NumPy · scikit-learn · matplotlib · seaborn
