"""Stage 5: column-wise missing-value strategies chosen from the missingness mechanism."""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

NUMERIC = ["weight_kg", "volume_m3", "freight_cost"]


def impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # MAR: never fabricate the outcome being measured - flag instead.
    df["delivered_flag"] = df["actual_delivery_ts"].notna().astype("int8")

    # MCAR with meaning: guest checkout is a real segment.
    df["customer_id"] = df["customer_id"].fillna("GUEST")

    # MAR: freight is priced by lane, carrier and weight band.
    df["weight_decile"] = pd.qcut(df["weight_kg"], 10, labels=False, duplicates="drop")
    grp = ["origin_dc", "carrier", "vehicle_type", "weight_decile"]
    df["freight_cost"] = df.groupby(grp)["freight_cost"].transform(lambda s: s.fillna(s.median()))
    df["freight_cost"] = df["freight_cost"].fillna(df["freight_cost"].median())

    # MNAR: dimension capture is skipped for small parcels - keep the signal.
    df["volume_imputed"] = df["volume_m3"].isna().astype("int8")
    df[NUMERIC] = IterativeImputer(random_state=42, max_iter=10).fit_transform(df[NUMERIC])
    df[NUMERIC] = KNNImputer(n_neighbors=5).fit_transform(df[NUMERIC])

    city_mode = (df.dropna(subset=["dest_pincode"])
                   .groupby("dest_city")["dest_pincode"]
                   .agg(lambda s: s.mode().iat[0]))
    df["dest_pincode"] = (df["dest_pincode"]
                          .fillna(df["dest_city"].map(city_mode))
                          .fillna("UNKNOWN"))
    return df
