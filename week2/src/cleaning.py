"""Stages 1-4: schema enforcement, deduplication, standardisation, business rules."""
import numpy as np
import pandas as pd

TS_COLS = ["order_ts", "dispatch_ts", "promised_delivery_ts", "actual_delivery_ts"]

CARRIER_MAP = {
    "bluedart": "BlueDart", "blue dart": "BlueDart", "bluedart ": "BlueDart",
    "bluedart india": "BlueDart", "delhivery": "Delhivery", "dhl": "DHL",
    "dhl express": "DHL", "own fleet": "OwnFleet", "ownfleet": "OwnFleet",
    "gati": "Gati", "safexpress": "Safexpress", "ekart": "Ekart",
    "xpressbees": "XpressBees", "ecom express": "EcomExpress",
}

RULES = {
    "delivery_before_dispatch": lambda d: d["actual_delivery_ts"] < d["dispatch_ts"],
    "dispatch_before_order": lambda d: d["dispatch_ts"] < d["order_ts"],
    "nonpositive_weight": lambda d: d["weight_kg"] <= 0,
    "nonpositive_freight": lambda d: d["freight_cost"] <= 0,
    "transit_over_30_days": lambda d: (d["actual_delivery_ts"] - d["dispatch_ts"]).dt.days > 30,
}


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """errors='coerce' routes unparseable values into the missing-data pathway."""
    df = df.copy()
    money = df["freight_cost"].astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    df["freight_cost"] = pd.to_numeric(money, errors="coerce")
    for col in TS_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True, utc=True)
    df["dest_pincode"] = (df["dest_pincode"].astype("string")
                          .str.extract(r"(\d{6})")[0].str.zfill(6))
    for col in ["weight_kg", "volume_m3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve on the business key, keeping the most complete and most recent row."""
    df = df.assign(_completeness=df.notna().sum(axis=1))
    df = (df.sort_values(["order_id", "_completeness", "order_ts"],
                         ascending=[True, False, False])
            .drop_duplicates(subset=["order_id"], keep="first")
            .drop(columns="_completeness"))
    return df.reset_index(drop=True)


def standardise_text(df: pd.DataFrame) -> pd.DataFrame:
    def norm(s):
        return (s.astype("string").str.strip().str.lower()
                 .str.replace(r"[^a-z0-9 ]", "", regex=True)
                 .str.replace(r"\s+", " ", regex=True))

    df = df.copy()
    df["carrier"] = norm(df["carrier"]).map(CARRIER_MAP).fillna("Unknown")
    df["dest_city"] = norm(df["dest_city"]).str.title()
    df["origin_dc"] = norm(df["origin_dc"]).str.upper()
    df["vehicle_type"] = norm(df["vehicle_type"]).str.title()
    return df


def apply_rules(df: pd.DataFrame, quarantine_path: str | None = None):
    """Violations are quarantined and reported, never silently deleted."""
    flags = pd.DataFrame(index=df.index)
    for name, rule in RULES.items():
        flags[name] = rule(df).fillna(False)
    bad = flags.any(axis=1)
    rejects = df[bad].join(flags[bad])
    if quarantine_path:
        rejects.to_parquet(quarantine_path)
    print(flags.sum().to_string())
    return df[~bad].reset_index(drop=True), rejects
