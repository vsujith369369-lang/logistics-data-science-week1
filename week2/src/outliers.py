"""Stage 6: separate data-entry errors from legitimate business extremes."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DOMAIN_LIMITS = {"weight_kg": (0.01, 25000), "volume_m3": (0.001, 90),
                 "transit_hours": (0.25, 720)}
WINSORIZE = ["freight_cost", "weight_kg", "transit_hours"]
ISO_FEATURES = ["weight_kg", "volume_m3", "distance_km", "freight_cost"]


def modified_zscore(s: pd.Series) -> pd.Series:
    """MAD-based; the outliers do not inflate the dispersion estimate."""
    med = s.median()
    mad = (s - med).abs().median()
    return 0.6745 * (s - med) / mad if mad else pd.Series(0.0, index=s.index)


def iqr_bounds(s: pd.Series, k: float = 1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def treat_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in DOMAIN_LIMITS.items():
        if col in df:
            df.loc[~df[col].between(lo, hi), col] = np.nan

    df["freight_per_kg"] = df["freight_cost"] / df["weight_kg"]
    z = df.groupby("origin_dc")["freight_per_kg"].transform(modified_zscore)
    df.loc[z.abs() > 3.5, "freight_cost"] = np.nan  # re-imputed downstream

    for col in WINSORIZE:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[f"{col}_wins"] = df[col].clip(lo, hi)  # clean-for-modelling copy

    iso = IsolationForest(contamination=0.01, random_state=42)
    df["anomaly_flag"] = (iso.fit_predict(df[ISO_FEATURES].fillna(0)) == -1).astype("int8")
    return df
