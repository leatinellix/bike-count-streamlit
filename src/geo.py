from __future__ import annotations
from typing import Callable, Optional, Tuple
import os
import pandas as pd
import numpy as np

STD_ID, STD_NAME = "site_id", "site_name"

# ---------- Parsing helpers ----------

def parse_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exposes numeric 'lat'/'lon' if present or derivable from a 'coordinates' column.
    Accepted forms for coordinates: "48.86,2.35" or "POINT (2.35 48.86)".
    Does not raise on failure; just returns df with best-effort lat/lon columns.
    """
    out = df.copy()

    # Already present?
    lat_cols = [c for c in out.columns if c.lower() in {"lat", "latitude"}]
    lon_cols = [c for c in out.columns if c.lower() in {"lon", "lng", "longitude"}]
    if lat_cols and lon_cols:
        out["lat"] = pd.to_numeric(out[lat_cols[0]], errors="coerce")
        out["lon"] = pd.to_numeric(out[lon_cols[0]], errors="coerce")
        return out

    # Try from a "coordinates" column
    if "coordinates" in out.columns:
        s = out["coordinates"].astype(str)

        # WKT pattern: "POINT (lon lat)"
        w = s.str.extract(r"POINT\s*\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)", expand=True)
        if not w.isna().all().all():
            out["lon"] = pd.to_numeric(w[0], errors="coerce")
            out["lat"] = pd.to_numeric(w[1], errors="coerce")

        # CSV-like: "a,b" (try to infer ordering)
        sp = s.str.extract(r"([-\d\.]+)\s*,\s*([-\d\.]+)", expand=True)
        if not sp.isna().all().all():
            a = pd.to_numeric(sp[0], errors="coerce")
            b = pd.to_numeric(sp[1], errors="coerce")
            lat_first = (a.between(-90, 90) & b.between(-180, 180))
            lon_first = (a.between(-180, 180) & b.between(-90, 90))
            out.loc[lat_first, ["lat", "lon"]] = np.c_[a[lat_first], b[lat_first]]
            out.loc[lon_first, ["lon", "lat"]] = np.c_[a[lon_first], b[lon_first]]

    return out

# ---------- Coord store (csv) ----------

def normalize_coord_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns are exactly: site_id, site_name (optional), lat, lon."""
    cols = {c.lower(): c for c in df.columns}
    out = df.copy()
    def _ensure(name):
        if name in out.columns:
            return
        # try case-insensitive match
        m = [c for c in out.columns if c.lower() == name]
        if m:
            out.rename(columns={m[0]: name}, inplace=True)
    for col in [STD_ID, STD_NAME, "lat", "lon"]:
        _ensure(col)
    return out

def load_site_coords(path: str = "external_data/site_coords.csv") -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None
    df = normalize_coord_columns(df)
    # keep only the expected columns if available
    keep = [c for c in [STD_ID, STD_NAME, "lat", "lon"] if c in df.columns]
    return df[keep]

def save_site_coords(df: pd.DataFrame, path: str = "external_data/site_coords.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = normalize_coord_columns(df)
    df.to_csv(path, index=False)

# ---------- Geocoding (pluggable) ----------

def make_nominatim_geocode_fn(user_agent: str = "bike-counts-paris"):
    """
    Returns a callable geocode_fn(query: str) -> Optional[Tuple[lat, lon]].
    Uses geopy.Nominatim under the hood. Import is local to avoid hard dep in tests.
    """
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent=user_agent)
    def _fn(query: str) -> Optional[Tuple[float, float]]:
        try:
            loc = geolocator.geocode(query, timeout=10)
            if not loc:
                return None
            return (float(loc.latitude), float(loc.longitude))
        except Exception:
            return None
    return _fn

def geocode_site_names(
    stations_df: pd.DataFrame,
    geocode_fn: Callable[[str], Optional[Tuple[float, float]]],
    site_id_col: str = STD_ID,
    site_name_col: str = STD_NAME,
    city_suffix: str = "Paris, France",
) -> pd.DataFrame:
    """
    Build a coordinates dataframe from unique station rows using a provided geocode function.
    No network call assumptions; pass a fake geocode_fn in tests.
    """
    need = [c for c in [site_id_col, site_name_col] if c in stations_df.columns]
    if not need:
        raise ValueError("Need at least 'site_id' or 'site_name' in stations_df")

    rows = []
    for _, r in stations_df[need].drop_duplicates().iterrows():
        name = r.get(site_name_col) if site_name_col in stations_df.columns else None
        query = f"{name}, {city_suffix}" if pd.notna(name) else city_suffix
        latlon = geocode_fn(query)
        lat = latlon[0] if latlon else None
        lon = latlon[1] if latlon else None
        row = {}
        if site_id_col in r.index:
            row[STD_ID] = r[site_id_col]
        if site_name_col in r.index:
            row[STD_NAME] = r[site_name_col]
        row.update({"lat": lat, "lon": lon})
        rows.append(row)
    out = pd.DataFrame(rows)
    return out

# ---------- Merge logic ----------

def merge_coords(
    base_df: pd.DataFrame,
    coords_df: pd.DataFrame,
    site_id_col: str = STD_ID,
    site_name_col: str = STD_NAME,
) -> pd.DataFrame:
    """
    Merge coordinates into base_df, first by site_id then fill missing via site_name.
    """
    out = base_df.copy()
    cdf = normalize_coord_columns(coords_df)

    if site_id_col in out.columns and site_id_col in cdf.columns:
        out = out.merge(cdf[[site_id_col, "lat", "lon"]], on=site_id_col, how="left")

    need_fill = out["lat"].isna() if "lat" in out.columns else pd.Series(True, index=out.index)
    if need_fill.any() and site_name_col in out.columns and site_name_col in cdf.columns:
        out = out.merge(
            cdf[[site_name_col, "lat", "lon"]].rename(columns={"lat": "lat_by_name", "lon": "lon_by_name"}),
            on=site_name_col, how="left"
        )
        out["lat"] = out.get("lat", pd.Series(index=out.index, dtype="float64")).fillna(out.get("lat_by_name"))
        out["lon"] = out.get("lon", pd.Series(index=out.index, dtype="float64")).fillna(out.get("lon_by_name"))
        drop = [c for c in ["lat_by_name", "lon_by_name"] if c in out.columns]
        if drop: out.drop(columns=drop, inplace=True)
    return out
