"""Week 2 - multi-source ingestion for the NorthStar Logistics dataset."""
import glob
import json

import pandas as pd

DTYPES = {
    "order_id": "string", "customer_id": "string", "origin_dc": "string",
    "dest_city": "string", "dest_pincode": "string", "carrier": "string",
    "vehicle_type": "string", "freight_cost": "string",
}


def load_wms(path: str) -> pd.DataFrame:
    """Nightly CSV batches from the warehouse management system."""
    frames = [pd.read_csv(f, dtype=DTYPES) for f in glob.glob(f"{path}/wms_*.csv")]
    return pd.concat(frames, ignore_index=True)


def load_tms(path: str) -> pd.DataFrame:
    """Hourly JSON pull from the transport management system."""
    with open(path) as fh:
        payload = json.load(fh)
    return pd.json_normalize(payload["shipments"])


def load_cost_ledger(path: str) -> pd.DataFrame:
    """Weekly Excel workbook from finance; two merged header rows are skipped."""
    return pd.read_excel(path, sheet_name="freight", skiprows=2)


def load_raw(base: str = "data/raw") -> pd.DataFrame:
    raw = (
        load_wms(base)
        .merge(load_tms(f"{base}/tms.json"), on="order_id", how="left")
        .merge(load_cost_ledger(f"{base}/ledger.xlsx"), on="order_id", how="left")
    )
    print(raw.shape, round(raw.memory_usage(deep=True).sum() / 1e6, 1), "MB")
    return raw
