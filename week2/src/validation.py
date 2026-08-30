"""Stage 9: sixteen automated assertions and the data quality scorecard."""
import numpy as np
import pandas as pd
from scipy import stats

VALID_CARRIERS = {"BlueDart", "Delhivery", "DHL", "OwnFleet", "Gati",
                  "Safexpress", "Ekart", "XpressBees", "EcomExpress", "Unknown"}
VALID_DCS = {"DC-NORTH", "DC-WEST", "DC-SOUTH"}
MODEL_FEATURES = ["weight_kg", "volume_m3", "freight_cost", "distance_km",
                  "capacity_utilisation", "cost_per_km"]


def validate(df: pd.DataFrame, raw_rows: int, start: str, end: str) -> pd.Series:
    checks = {
        "no_duplicate_orders": df["order_id"].is_unique,
        "no_null_keys": df["order_id"].notna().all(),
        "no_null_model_features": df[MODEL_FEATURES].notna().all().all(),
        "weight_positive": (df["weight_kg"] > 0).all(),
        "freight_positive": (df["freight_cost"] > 0).all(),
        "transit_nonneg": (df["transit_hours"].dropna() >= 0).all(),
        "dispatch_after_order": (df["dispatch_ts"] >= df["order_ts"]).all(),
        "carrier_in_domain": df["carrier"].isin(VALID_CARRIERS).all(),
        "dc_in_domain": df["origin_dc"].isin(VALID_DCS).all(),
        "pincode_format": df["dest_pincode"].str.match(r"^(\d{6}|UNKNOWN)$").all(),
        "utilisation_bounded": df["capacity_utilisation"].between(0, 1.5).all(),
        "row_retention": len(df) / raw_rows > 0.90,
        "date_range_valid": df["order_ts"].between(start, end).all(),
        "otd_plausible": 0.5 < df["is_on_time"].mean() < 1.0,
        "no_constant_columns": (df.nunique() > 1).all(),
        "skew_controlled": abs(stats.skew(np.log1p(df["freight_cost"]))) < 1.0,
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise AssertionError(f"Validation failed: {failed}")
    return pd.Series(checks)


def scorecard(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"metric": "rows", "raw": len(raw), "clean": len(clean)},
        {"metric": "columns", "raw": raw.shape[1], "clean": clean.shape[1]},
        {"metric": "null_cells", "raw": int(raw.isna().sum().sum()),
         "clean": int(clean.isna().sum().sum())},
        {"metric": "duplicate_orders", "raw": int(raw.duplicated(subset=["order_id"]).sum()),
         "clean": int(clean.duplicated(subset=["order_id"]).sum())},
    ])
