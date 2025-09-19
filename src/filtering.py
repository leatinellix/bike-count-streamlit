# src/filtering.py
from typing import Iterable, Optional
import pandas as pd

def filter_by_date_range(df: pd.DataFrame, start: Optional[pd.Timestamp] = None,
                         end: Optional[pd.Timestamp] = None, datetime_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if start is not None:
        out = out[out[datetime_col] >= start]
    if end is not None:
        out = out[out[datetime_col] <= end]
    return out

def filter_by_site(df: pd.DataFrame, sites: Optional[Iterable] = None, station_col: str = "site_id") -> pd.DataFrame:
    if not sites:
        return df.copy()
    sites = {str(s) for s in sites}
    return df[df[station_col].astype(str).isin(sites)].copy()

def apply_filters(df: pd.DataFrame, start=None, end=None, sites=None,
                  datetime_col: str = "date", station_col: str = "site_id") -> pd.DataFrame:
    out = filter_by_date_range(df, start=start, end=end, datetime_col=datetime_col)
    out = filter_by_site(out, sites=sites, station_col=station_col)
    return out
