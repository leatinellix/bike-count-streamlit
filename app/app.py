import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from src.data_io import load_dataframe, ensure_required_columns
from src.filtering import apply_filters
from src.geo import parse_latlon, load_site_coords, merge_coords

st.set_page_config(page_title="Paris Bike Counts — Dashboard", layout="wide")

# ---- make KPI numbers smaller so everything fits ----
st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 28px !important; }
div[data-testid="stMetricLabel"] { font-size: 14px !important; }
</style>
""", unsafe_allow_html=True)

CFG = {
    "paths": {"train": "data/train.parquet"},
    "cols": {"dt": "date", "y": "log_bike_count", "site": "site_id"},
}

@st.cache_data(show_spinner=False)
def get_train() -> pd.DataFrame:
    df = load_dataframe(CFG["paths"]["train"], datetime_col=CFG["cols"]["dt"])
    df = ensure_required_columns(df, [CFG["cols"]["dt"], CFG["cols"]["site"]])
    df[CFG["cols"]["dt"]] = pd.to_datetime(df[CFG["cols"]["dt"]], errors="coerce")
    if CFG["cols"]["y"] not in df.columns and "bike_count" in df.columns:
        df[CFG["cols"]["y"]] = np.log1p(df["bike_count"])
    return df

# ---------------- Sidebar ----------------
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

# --------------- Filter data ---------------
start = pd.Timestamp(date_range[0])
end   = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
df = apply_filters(df_all, start=start, end=end, sites=selected_sites,
                   datetime_col=dt, station_col=site)

# --------------- KPIs ---------------
k1, k2, k3 = st.columns(3)
k1.metric("Rows", f"{len(df):,}")
k2.metric("Unique stations", f"{df[site].nunique():,}")
k3.metric("Period", f"{start.date()} → {end.date()}")

# --------------- Tabs ---------------
tab_map, tab_series, tab_table = st.tabs(["Map", "Time series", "Table"])

# ---- Map tab ----
with tab_map:
    coords = load_site_coords()

    # aggregate counts per station on raw scale
    if "bike_count" in df.columns:
        agg = df.groupby(site, as_index=False)["bike_count"].sum().rename(columns={"bike_count": "count"})
    else:
        tmp = df.copy(); tmp["count"] = np.expm1(tmp[y])
        agg = tmp.groupby(site, as_index=False)["count"].sum()

    first_rows = df.sort_values(dt).drop_duplicates(subset=[site], keep="first")
    base_cols = [site] + (["site_name"] if "site_name" in first_rows.columns else []) + (["coordinates"] if "coordinates" in first_rows.columns else [])
    map_df = first_rows[base_cols].merge(agg, on=site, how="left")

    # derive lat/lon from in-data coordinates if present, then merge cache
    map_df = parse_latlon(map_df)
    if coords is not None:
        map_df = merge_coords(map_df, coords)

    if {"lat","lon"}.issubset(map_df.columns) and map_df["lat"].notna().any():
        import pydeck as pdk
        view_state = pdk.ViewState(latitude=48.8566, longitude=2.3522, zoom=11, pitch=0)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df.dropna(subset=["lat","lon"]),
            get_position='[lon, lat]',
            get_radius="count * 2",
            radius_min_pixels=2, radius_max_pixels=25,
            get_fill_color=[37, 99, 235, 160],
            pickable=True,
        )
        has_name = "site_name" in map_df.columns
        tooltip = {"html": "<b>site_id:</b> {site_id}" + ( "<br/><b>site_name:</b> {site_name}" if has_name else "" ) + "<br/><b>total count:</b> {count}"}
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip), use_container_width=True)
        st.caption("Circle size is proportional to total bike counts over the selected period.")
    else:
        st.info("No coordinates found. Provide lat/lon in your data, or add a cache at external_data/site_coords.csv.")
        st.dataframe(map_df.head())

# ---- Time series tab (clearer: smoothing + average line) ----
with tab_series:
    left, right = st.columns([2, 1])
    with right:
        show_avg = st.checkbox("Show average across stations", value=True)
        smooth = st.slider("Smoothing window (periods)", min_value=1, max_value=24, value=6, help="Applies a rolling mean per station.")
    with left:
        st.caption("Use controls on the right to smooth and optionally show the average.")

    base = df[[dt, y, site]].copy()
    if aggregation == "Daily":
        base = base.set_index(dt).groupby(site)[y].resample("D").mean().reset_index()
    elif aggregation == "Weekly":
        base = base.set_index(dt).groupby(site)[y].resample("W").mean().reset_index()
    else:
        base = base.sort_values(dt)

    base = base.rename(columns={dt: "date", y: "value", site: "station"})

    if smooth > 1:
        base["value"] = base.groupby("station")["value"].transform(lambda s: s.rolling(smooth, min_periods=1).mean())

    lines = alt.Chart(base).mark_line(opacity=0.6, strokeWidth=2).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="log_bike_count"),
        color=alt.Color("station:N", legend=alt.Legend(title="Station")),
        tooltip=[alt.Tooltip("date:T"), "station", alt.Tooltip("value:Q", format=".3f")],
    )

    if show_avg:
        avg = base.groupby("date", as_index=False)["value"].mean()
        avg_layer = alt.Chart(avg).mark_line(strokeWidth=3).encode(
            x="date:T", y="value:Q", color=alt.value("#111111")
        )
        chart = (lines + avg_layer).properties(height=360).interactive()
    else:
        chart = lines.properties(height=360).interactive()

    st.altair_chart(chart, use_container_width=True)

# ---- Table tab ----
with tab_table:
    st.dataframe(df.head(500), use_container_width=True)

