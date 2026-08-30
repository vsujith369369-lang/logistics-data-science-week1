"""Stage 7: turn raw operational fields into analytically meaningful features."""
import numpy as np
import pandas as pd

CAPACITY_MAP = {"Tata Ace": 750, "Eicher 14Ft": 3500, "Eicher 19Ft": 7500,
                "Truck 32Ft": 16000, "Bike": 40}


def derive_features(df: pd.DataFrame, capacity_map=None, festival_dates=None) -> pd.DataFrame:
    capacity_map = capacity_map or CAPACITY_MAP
    festival_dates = pd.to_datetime(pd.Series(festival_dates or [], dtype="datetime64[ns]"))
    df = df.copy()

    df["transit_hours"] = (df["actual_delivery_ts"] - df["dispatch_ts"]).dt.total_seconds() / 3600
    df["order_to_dispatch_hours"] = (df["dispatch_ts"] - df["order_ts"]).dt.total_seconds() / 3600
    df["is_on_time"] = (df["actual_delivery_ts"] <= df["promised_delivery_ts"]).astype("Int8")
    df["delay_hours"] = (((df["actual_delivery_ts"] - df["promised_delivery_ts"])
                          .dt.total_seconds() / 3600).clip(lower=0))

    df["cost_per_km"] = df["freight_cost"] / df["distance_km"].replace(0, np.nan)
    df["chargeable_weight"] = np.maximum(df["weight_kg"], df["volume_m3"] * 167)
    df["capacity_utilisation"] = df["chargeable_weight"] / df["vehicle_type"].map(capacity_map)

    df["dow"] = df["order_ts"].dt.dayofweek
    df["hour"] = df["order_ts"].dt.hour
    df["is_month_end"] = df["order_ts"].dt.is_month_end.astype("int8")
    df["is_festival"] = df["order_ts"].dt.tz_localize(None).dt.normalize().isin(festival_dates).astype("int8")
    return df
