"""Phase 4-5: SKU-level demand forecasting and inventory policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

FESTIVAL_WEEKS: set[int] = {41, 42, 43, 44, 51, 52}


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag, rolling and calendar features on the SKU-week grain."""
    df = df.sort_values(["sku", "week"]).copy()
    g = df.groupby("sku")["quantity"]

    for lag in (1, 2, 4, 52):
        df[f"lag_{lag}"] = g.shift(lag)
    df["roll_4"] = g.shift(1).rolling(4).mean().reset_index(level=0, drop=True)
    df["roll_12"] = g.shift(1).rolling(12).mean().reset_index(level=0, drop=True)

    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
    df["month"] = df["week"].dt.month
    df["is_festival"] = df["week_of_year"].isin(FESTIVAL_WEEKS).astype(int)
    return df.dropna()


def train_forecaster(feat: pd.DataFrame, holdout_weeks: int = 12):
    """Walk-forward split, LightGBM model, MAPE/RMSE/bias evaluation."""
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

    cut = feat["week"].max() - pd.Timedelta(weeks=holdout_weeks)
    train, test = feat[feat["week"] <= cut], feat[feat["week"] > cut]

    features = [
        c for c in feat.columns
        if c.startswith(("lag_", "roll_"))
        or c in ("week_of_year", "month", "is_festival")
    ]

    model = LGBMRegressor(
        n_estimators=800, learning_rate=0.05, num_leaves=63, random_state=42
    )
    model.fit(train[features], train["quantity"])
    pred = model.predict(test[features])

    metrics = {
        "mape": float(mean_absolute_percentage_error(test["quantity"], pred)),
        "rmse": float(np.sqrt(mean_squared_error(test["quantity"], pred))),
        "bias": float(np.mean(pred - test["quantity"])),
    }
    return model, features, metrics


def seasonal_naive_baseline(feat: pd.DataFrame) -> float:
    """Benchmark every model against last year's same week."""
    from sklearn.metrics import mean_absolute_percentage_error

    valid = feat.dropna(subset=["lag_52"])
    return float(mean_absolute_percentage_error(valid["quantity"], valid["lag_52"]))


def reorder_point(
    mean_daily_demand: float,
    sigma_demand: float,
    lead_time_days: float,
    service_level: float = 0.95,
) -> tuple[float, float]:
    """Reorder point and safety stock at a target cycle service level."""
    z = norm.ppf(service_level)
    safety_stock = z * sigma_demand * np.sqrt(lead_time_days)
    return mean_daily_demand * lead_time_days + safety_stock, safety_stock


def eoq(annual_demand: float, order_cost: float, holding_cost_per_unit: float) -> float:
    """Economic order quantity."""
    return float(np.sqrt(2 * annual_demand * order_cost / holding_cost_per_unit))
