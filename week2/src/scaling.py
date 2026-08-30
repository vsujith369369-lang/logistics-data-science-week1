"""Stage 8: normalization, scaling and encoding - fitted on the training split only."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder,
                                   RobustScaler, StandardScaler)

log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        # right-skewed: compress then scale robustly
        ("skewed", Pipeline([("log", log1p), ("rob", RobustScaler())]),
         ["freight_cost", "weight_kg", "chargeable_weight"]),
        # approximately symmetric after winsorizing
        ("symmetric", StandardScaler(), ["distance_km", "transit_hours", "cost_per_km"]),
        # bounded with a known natural range
        ("bounded", MinMaxScaler(), ["capacity_utilisation"]),
        # low-cardinality nominal
        ("nominal", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=0.01, sparse_output=False),
         ["origin_dc", "vehicle_type", "carrier"]),
    ], remainder="drop", verbose_feature_names_out=False)


def cyclical(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """Preserves adjacency of hour 23 and hour 0, which integer encoding destroys."""
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df.drop(columns=col)


def chronological_split(df: pd.DataFrame, cutoff: str = "2025-07-01"):
    """Temporal data demands a temporal split; a random split leaks the future."""
    train = df[df["order_ts"] < cutoff]
    test = df[df["order_ts"] >= cutoff]
    return train, test
