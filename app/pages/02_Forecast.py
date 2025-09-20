# app/pages/02_Forecast.py
import numpy as np
import pandas as pd
import streamlit as st

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from src.data_io import load_dataframe, ensure_required_columns
from src.modeling import build_pipeline, train, predict, DT, Y, SITE
from src.optimize import optimize_xgb



st.set_page_config(page_title="Bike Paris — Forecast", layout="wide")
st.title("Forecast")

TRAIN_PATH = "data/train.parquet"
TEST_PATH  = "data/final_test.parquet"

def mae_real_scale(y_true_log, y_pred_log):
    import numpy as np
    y_true = np.expm1(y_true_log)  # inverse de log1p
    y_pred = np.expm1(y_pred_log)
    return float(np.mean(np.abs(y_true - y_pred)))


@st.cache_data(show_spinner=False)
def get_train() -> pd.DataFrame:
    df = load_dataframe(TRAIN_PATH, datetime_col=DT)
    df = ensure_required_columns(df, [DT, Y])  # SITE peut (éventuellement) manquer
    # cast minimal propre
    df[DT] = pd.to_datetime(df[DT], errors="coerce")
    if SITE in df.columns:
        df[SITE] = df[SITE].astype(str)
    return df

@st.cache_data(show_spinner=False)
def get_final_test() -> pd.DataFrame:
    df = load_dataframe(TEST_PATH, datetime_col=DT)
    df[DT] = pd.to_datetime(df[DT], errors="coerce")
    if SITE in df.columns:
        df[SITE] = df[SITE].astype(str)
    return df

# --- UI ---
st.subheader("Données")
colA, colB = st.columns(2)
with colA:
    src = st.radio("Source des features à prédire", ["Subset du train (démo)", "final_test.parquet"], horizontal=False)
with colB:
    n_estimators = st.slider("n_estimators (XGB)", min_value=100, max_value=600, value=200, step=50)

# Chargement train
try:
    df_train = get_train()
except Exception as e:
    st.error(f"Impossible to load {TRAIN_PATH}: {e}")
    st.stop()

# Optionnel: subset pour la démo (évite d'entraîner/afficher sur trop gros)
st.caption(f"Train: {len(df_train):,} lines")

# Sidebar (place this ABOVE the Train/Predict buttons)
DEFAULT_CUTOFF = pd.Timestamp("2021-05-01")

cutoff = st.date_input(
    "Time split cutoff (train ≤ cutoff < valid)",
    value=DEFAULT_CUTOFF.date()
)
cutoff_ts = pd.Timestamp(cutoff)

st.markdown("### Hyperparameter search (Optuna)")
colO1, colO2, colO3 = st.columns([1,1,2])
with colO1:
    n_trials = st.slider("n_trials", 10, 100, 30, 5)
with colO2:
    cutoff_opt = st.date_input("Cutoff", value=pd.to_datetime("2021-05-01"))
with colO3:
    st.caption("Recherche = +lent. Utilise un split temporel pour éviter la fuite.")

if st.button("Run Optuna"):
    from src.optimize import optimize_xgb
    with st.spinner("Optuna loading…"):
        best_params, best_mae = optimize_xgb(df_train, pd.Timestamp(cutoff_opt), n_trials=n_trials)
    st.success(f"Best MAE(log): {best_mae:.3f}")
    st.json(best_params)
    st.session_state["best_params"] = best_params


# Entraînement
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
    st.success(f"Model trained ✅ — train: {len(train_df):,} | valid: {len(valid_df):,}")



# Prédiction
if st.button("Predict"):
    if "model" not in st.session_state:
        st.error("First train the model.")
    else:
        model = st.session_state["model"]

        if src == "Subset":
            # On prend une petite fenêtre récente pour visualiser
            df_sub = df_train.sort_values(DT).tail(200).copy()
            y_pred = predict(model, df_sub)
            out = df_sub[[DT]].copy()
            out["y_pred"] = y_pred
            if Y in df_sub.columns:
                out["y_true"] = df_sub[Y].values

            st.subheader("Observed vs predicted (subset train)")
            st.dataframe(out.head(30))
            cols = ["y_pred"] + (["y_true"] if "y_true" in out.columns else [])
            st.line_chart(out.set_index(DT)[cols])

            # petite métrique
            if "y_true" in out.columns:
                mae_log = float(np.mean(np.abs(out["y_true"] - out["y_pred"])))
                mae_count = mae_real_scale(out["y_true"], out["y_pred"])
                c1, c2 = st.columns(2)
                c1.metric("MAE (log)", f"{mae_log:.3f}")
                c2.metric("MAE (comptes)", f"{mae_count:,.0f}")


        else:
            # Prédire sur final_test
            try:
                df_test = get_final_test()
            except Exception as e:
                st.error(f"Impossible to load {TEST_PATH}: {e}")
            else:
                y_pred = predict(model, df_test)
                sub = pd.DataFrame({"log_bike_count": y_pred})
                sub.index.name = "Id"

                st.subheader("Submission preview")
                st.dataframe(sub.head(20))

                csv = sub.to_csv().encode("utf-8")
                st.download_button("Download CSV",
                                   data=csv, file_name="kaggle_submission.csv", mime="text/csv")
                st.success(f"{len(sub):,} lines ready for submission.")
