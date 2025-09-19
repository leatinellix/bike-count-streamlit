import yaml, pandas as pd, streamlit as st
from src.data_io import load_dataframe, ensure_required_columns
from src.filtering import apply_filters
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


st.set_page_config(page_title="Bike Paris — Overview", layout="wide")

CFG = {"paths":{"train":"data/train.parquet"},
       "cols":{"dt":"date","y":"log_bike_count","site":"site_id"},
       "tz":"Europe/Paris"}

@st.cache_data(show_spinner=False)
def get_train():
    # charge
    df = load_dataframe(CFG["paths"]["train"], datetime_col=CFG["cols"]["dt"])
    # colonnes requises
    df = ensure_required_columns(df, [CFG["cols"]["dt"], CFG["cols"]["y"], CFG["cols"]["site"]])
    # PARSE SANS TZ (IMPORTANT)
    df[CFG["cols"]["dt"]] = pd.to_datetime(df[CFG["cols"]["dt"]], errors="coerce")
    return df


st.title("Bike Count Paris — Overview")
try:
    df = get_train()
except Exception as e:
    st.error(f"Impossible de charger {CFG['paths']['train']} — {e}")
    st.stop()

with st.sidebar:
    st.header("Filtres")
    dmin, dmax = df[CFG["cols"]["dt"]].min().date(), df[CFG["cols"]["dt"]].max().date()
    period = st.date_input("Période", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    sites = sorted(df[CFG["cols"]["site"]].astype(str).unique())
    sel = st.multiselect("Stations (site_id)", options=sites, default=sites[:5])

start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
filt = apply_filters(df, start=start, end=end, sites=sel,
                     datetime_col=CFG["cols"]["dt"], station_col=CFG["cols"]["site"])

c1,c2,c3 = st.columns(3)
c1.metric("Lignes", len(filt))
c2.metric("Stations uniques", filt[CFG["cols"]["site"]].nunique())
c3.metric("Période", f"{start.date()} → {end.date()}")

st.subheader("Aperçu")
st.dataframe(filt.head(100))

st.subheader("Série temporelle (log_bike_count)")
plot = filt[[CFG["cols"]["dt"], CFG["cols"]["y"]]].sort_values(CFG["cols"]["dt"]).set_index(CFG["cols"]["dt"])
st.line_chart(plot[CFG["cols"]["y"]])
