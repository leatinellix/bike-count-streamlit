import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from src.data_io import load_dataframe, ensure_required_columns
from src.modeling import build_pipeline, train, predict, DT, Y, SITE
from src.geo import load_site_coords, merge_coords, parse_latlon

st.set_page_config(page_title="Paris Bike Counts — Forecast", layout="wide")
st.title("Forecast")

TRAIN_PATH, TEST_PATH = "data/train.parquet", "data/final_test.parquet"

@st.cache_data(show_spinner=False)
def get_train() -> pd.DataFrame:
    df = load_dataframe(TRAIN_PATH, datetime_col=DT)
    df = ensure_required_columns(df, [c for c in [DT, SITE] if c in df.columns] or [DT])
    df[DT] = pd.to_datetime(df[DT], errors="coerce")
    if SITE in df.columns: df[SITE] = df[SITE].astype(str)
    if Y not in df.columns and "bike_count" in df.columns:
        df[Y] = np.log1p(df["bike_count"])
    return df

@st.cache_data(show_spinner=False)
def get_final_test() -> pd.DataFrame:
    df = load_dataframe(TEST_PATH, datetime_col=DT)
    df[DT] = pd.to_datetime(df[DT], errors="coerce")
    if SITE in df.columns: df[SITE] = df[SITE].astype(str)
    return df

def mae_counts(y_true_log, y_pred_log) -> float:
    return float(np.mean(np.abs(np.expm1(y_true_log) - np.expm1(y_pred_log))))

# ---------- Controls (sidebar) ----------
with st.sidebar:
    st.header("Settings")
    source = st.radio("Data source", ["Training & validation data", "Test data"], index=0)

    DEFAULT_CUTOFF = pd.Timestamp("2021-05-01")
    cutoff = st.date_input("Time split cutoff", value=DEFAULT_CUTOFF.date())
    cutoff_ts = pd.Timestamp(cutoff)

    with st.expander("Model", expanded=True):
        n_estimators = st.slider("n_estimators (XGB)", 100, 1200, 400, 50)

    with st.expander("Hyperparameter search (Optuna)"):
        n_trials = st.slider("n_trials", 10, 100, 30, 5)
        cutoff_opt = st.date_input("Optuna cutoff", value=DEFAULT_CUTOFF.date())
        if st.button("Run Optuna"):
            from src.optimize import optimize_xgb
            with st.spinner("Running Optuna…"):
                best_params, best_mae = optimize_xgb(get_train(), pd.Timestamp(cutoff_opt), n_trials=n_trials)
            st.success(f"Best MAE (log): {best_mae:.3f}")
            st.json(best_params, expanded=False)
            st.session_state["best_params"] = best_params

# ---------- Load data ----------
try:
    df_train = get_train()
except Exception as e:
    st.error(f"Cannot load train data: {e}")
    st.stop()

st.caption(f"Train: {len(df_train):,} rows")

# ---------- Actions ----------
c1, c2 = st.columns([1,1])

with c1:
    if st.button("Train model", type="primary"):
        params = st.session_state.get("best_params", None)
        pipe = build_pipeline(params or {"n_estimators": n_estimators})
        df_sorted = df_train.sort_values(DT)
        train_df = df_sorted[df_sorted[DT] <= cutoff_ts].copy()
        valid_df = df_sorted[df_sorted[DT] >  cutoff_ts].copy()
        y_train = train_df[Y]
        model = pipe.fit(train_df.drop(columns=[Y]), y_train)
        st.session_state["model"] = model
        st.session_state["valid_df"] = valid_df
        st.success(f"Model trained — train: {len(train_df):,} | valid: {len(valid_df):,}")

with c2:
    if st.button("Predict"):
        if "model" not in st.session_state:
            st.error("Please train the model first.")
        else:
            model = st.session_state["model"]

            # Validation metrics (if ground-truth exists)
            if "valid_df" in st.session_state and len(st.session_state["valid_df"]) > 0:
                vdf = st.session_state["valid_df"]
                y_pred_v = predict(model, vdf)
                if Y in vdf.columns:
                    y_true_v = vdf[Y].values
                    mae_log = float(np.mean(np.abs(y_true_v - y_pred_v)))
                    mae_cnt = mae_counts(y_true_v, y_pred_v)
                    st.info(f"Validation MAE — log: {mae_log:.3f} | counts: {mae_cnt:,.0f}")

            # Choose source to predict
            if source == "Training & validation data":
                df_sub = df_train.sort_values(DT).tail(200).copy()
                y_pred = predict(model, df_sub)
                out = df_sub[[DT, SITE] if SITE in df_sub.columns else [DT]].copy()
                out["y_pred"] = y_pred
                if Y in df_sub.columns: out["y_true"] = df_sub[Y].values

                st.subheader("Observed vs predicted (subset)")
                st.dataframe(out.head(30), use_container_width=True)

                plot_df = out.melt(id_vars=[DT],
                                   value_vars=[c for c in ["y_pred","y_true"] if c in out.columns],
                                   var_name="series", value_name="value")
                chart = (
                    alt.Chart(plot_df)
                    .mark_line(strokeWidth=2)
                    .encode(
                        x=alt.X(f"{DT}:T", title="Date"),
                        y=alt.Y("value:Q", title="log_bike_count"),
                        color=alt.Color("series:N", legend=alt.Legend(title="")),
                        tooltip=[DT, "series", alt.Tooltip("value:Q", format=".3f")],
                    )
                    .properties(height=340)
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)

            else:  # "Test data"
                try:
                    df_test = get_final_test()
                except Exception as e:
                    st.error(f"Cannot load test data: {e}")
                else:
                    y_pred = predict(model, df_test)
                    sub = pd.DataFrame({"log_bike_count": y_pred}); sub.index.name = "Id"
                    st.subheader("Submission preview")
                    st.dataframe(sub.head(20), use_container_width=True)
                    st.download_button("Download kaggle_submission.csv",
                                       data=sub.to_csv().encode("utf-8"),
                                       file_name="kaggle_submission.csv",
                                       mime="text/csv")

                    # ---- Prediction map (aggregated over test period) ----
                    if SITE in df_test.columns:
                        st.subheader("Predictions map")
                        df_vis = df_test[[SITE, DT]].copy()
                        df_vis["pred_count"] = np.expm1(y_pred)

                        agg = df_vis.groupby(SITE, as_index=False)["pred_count"].sum()

                        base = df_test.sort_values(DT).drop_duplicates(SITE, keep="first")
                        base = base[[SITE] + ([ "site_name" ] if "site_name" in base.columns else []) + ([ "coordinates" ] if "coordinates" in base.columns else [])]
                        base = parse_latlon(base)
                        coords = load_site_coords()
                        if coords is not None:
                            base = merge_coords(base, coords)

                        map_df = base.merge(agg, on=SITE, how="left")

                        if {"lat","lon"}.issubset(map_df.columns) and map_df["lat"].notna().any():
                            import pydeck as pdk
                            # scale radius 5..30 px
                            r = map_df["pred_count"].fillna(0.0)
                            denom = (r.max() - r.min()) or 1.0
                            map_df["radius"] = 5 + 25 * (r - r.min()) / denom

                            view_state = pdk.ViewState(latitude=48.8566, longitude=2.3522, zoom=11, pitch=0)
                            layer = pdk.Layer(
                                "ScatterplotLayer",
                                data=map_df.dropna(subset=["lat","lon"]),
                                get_position='[lon, lat]',
                                get_radius="radius",
                                get_fill_color=[16, 185, 129, 170],  # teal-ish
                                pickable=True,
                            )
                            has_name = "site_name" in map_df.columns
                            tooltip = {"html": "<b>site_id:</b> {site_id}" + ( "<br/><b>site_name:</b> {site_name}" if has_name else "" ) + "<br/><b>predicted total:</b> {pred_count}"}
                            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip), use_container_width=True)
                            st.caption("Circle size is proportional to the total predicted bike counts over the test horizon.")
                        else:
                            st.info("No coordinates found for predictions map (need lat/lon via data or external_data/site_coords.csv).")