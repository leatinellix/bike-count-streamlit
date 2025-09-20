import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt

from src.data_io import load_dataframe, ensure_required_columns
from src.filtering import apply_filters

st.set_page_config(page_title="Paris Bike Counts — Dashboard", layout="wide")

CFG = {
    "paths": {"train": "data/train.parquet"},
    "cols": {"dt": "date", "y": "log_bike_count", "site": "site_id"},
}

# ----------------------- Helpers -----------------------
def parse_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to expose 'lat' and 'lon' columns for mapping.
    Accepted inputs:
      - columns named 'lat'/'latitude' and 'lon'/'lng'/'longitude'
      - a 'coordinates' column like '48.86,2.35' or 'POINT (2.35 48.86)'
    If not found, returns df unchanged (no lat/lon).
    """
    out = df.copy()

    # already present?
    cand_lat = [c for c in out.columns if c.lower() in {"lat","latitude"}]
    cand_lon = [c for c in out.columns if c.lower() in {"lon","lng","longitude"}]
    if cand_lat and cand_lon:
        out["lat"] = pd.to_numeric(out[cand_lat[0]], errors="coerce")
        out["lon"] = pd.to_numeric(out[cand_lon[0]], errors="coerce")
        return out

    # coordinates column variants
    if "coordinates" in out.columns:
        s = out["coordinates"].astype(str)

        # "POINT (lon lat)"
        is_wkt = s.str.contains("POINT", case=False, na=False)
        if is_wkt.any():
            w = s.str.extract(r"POINT\s*\(\s*([-\d\.]+)\s+([-\d\.]+)\s*\)", expand=True)
            out.loc[is_wkt, "lon"] = pd.to_numeric(w[0], errors="coerce")
            out.loc[is_wkt, "lat"] = pd.to_numeric(w[1], errors="coerce")

        # "lat,lon" or "lon,lat"
        split = s.str.extract(r"([-\d\.]+)\s*,\s*([-\d\.]+)", expand=True)
        if not split.isna().all().all():
            a = pd.to_numeric(split[0], errors="coerce")
            b = pd.to_numeric(split[1], errors="coerce")
            # guess ordering by range
            lat_guess = (a.between(-90, 90) & b.between(-180, 180))
            lon_guess = (a.between(-180, 180) & b.between(-90, 90))

            out.loc[lat_guess, "lat"] = a[lat_guess]
            out.loc[lat_guess, "lon"] = b[lat_guess]
            out.loc[lon_guess, "lon"] = a[lon_guess]
            out.loc[lon_guess, "lat"] = b[lon_guess]

    return out

@st.cache_data(show_spinner=False)
def get_train() -> pd.DataFrame:
    df = load_dataframe(CFG["paths"]["train"], datetime_col=CFG["cols"]["dt"])
    need = [CFG["cols"]["dt"], CFG["cols"]["site"]]
    df = ensure_required_columns(df, need)
    df[CFG["cols"]["dt"]] = pd.to_datetime(df[CFG["cols"]["dt"]], errors="coerce")
    # build log target if needed
    if CFG["cols"]["y"] not in df.columns and "bike_count" in df.columns:
        df[CFG["cols"]["y"]] = np.log1p(df["bike_count"])
    return df

# ----------------------- Sidebar -----------------------
with st.sidebar:
    st.header("Filters")
    df_all = get_train()
    dt, y, site = CFG["cols"]["dt"], CFG["cols"]["y"], CFG["cols"]["site"]

    dmin, dmax = df_all[dt].min().date(), df_all[dt].max().date()
    date_range = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)

    site_list = sorted(df_all[site].astype(str).unique())
    default_sites = site_list[: min(12, len(site_list))]
    selected_sites = st.multiselect("Stations (site_id)", options=site_list, default=default_sites)

    aggregation = st.radio("Aggregation", ["Hourly", "Daily", "Weekly"], index=1, horizontal=True)

    if st.button("Reset"):
        st.experimental_rerun()

st.title("Paris Bike Counts — Dashboard")

# ----------------------- Filter data -----------------------
start = pd.Timestamp(date_range[0])
end   = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

df = apply_filters(df_all, start=start, end=end, sites=selected_sites,
                   datetime_col=dt, station_col=site)

# ----------------------- KPIs -----------------------
k1, k2, k3 = st.columns(3)
k1.metric("Rows", f"{len(df):,}")
k2.metric("Unique stations", f"{df[site].nunique():,}")
k3.metric("Period", f"{start.date()} → {end.date()}")

# ----------------------- Tabs -----------------------
tab_map, tab_series, tab_table = st.tabs(["Map", "Time series", "Table"])

with tab_map:
    # aggregate total counts by station (raw scale when possible)
    if "bike_count" in df.columns:
        agg = df.groupby(site, as_index=False)["bike_count"].sum().rename(columns={"bike_count": "count"})
    else:
        agg = df.copy()
        agg["count"] = np.expm1(agg[y])
        agg = agg.groupby(site, as_index=False)["count"].sum()

    # keep one row per station (bring coordinates if available)
    first_rows = df.sort_values(dt).drop_duplicates(subset=[site], keep="first")
    map_df = first_rows[[site]].merge(agg, on=site, how="left")
    map_df = parse_latlon(map_df)

    if {"lat","lon"}.issubset(map_df.columns) and map_df["lat"].notna().any():
        # deck.gl map
        view_state = pdk.ViewState(latitude=48.8566, longitude=2.3522, zoom=11, pitch=0)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df.dropna(subset=["lat","lon"]),
            get_position='[lon, lat]',
            get_radius="count * 2",   # scale; tweak if bubbles too large/small
            radius_min_pixels=2,
            radius_max_pixels=25,
            get_fill_color=[37, 99, 235, 160],  # blue, semi-transparent
            pickable=True,
        )
        tooltip = {"html": "<b>site_id:</b> {site_id}<br/><b>total count:</b> {count}"}
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip), use_container_width=True)
        st.caption("Circle size is proportional to total bike counts over the selected period.")
    else:
        st.info("No coordinates found (lat/lon or coordinates column). Add them to visualize the map.")
        st.write(map_df.head())

with tab_series:
    base = df[[dt, y, site]].copy()
    if aggregation == "Daily":
        base = base.set_index(dt).groupby(site)[y].resample("D").mean().reset_index()
    elif aggregation == "Weekly":
        base = base.set_index(dt).groupby(site)[y].resample("W").mean().reset_index()
    else:
        base = base.sort_values(dt)

    base = base.rename(columns={dt: "date", y: "value", site: "station"})
    chart = (
        alt.Chart(base)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="log_bike_count"),
            color=alt.Color("station:N", legend=alt.Legend(title="Station")),
            tooltip=[alt.Tooltip("date:T"), "station", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

with tab_table:
    st.dataframe(df.head(500), use_container_width=True)

