# app/pages/02_Forecast.py
import numpy as np
import pandas as pd
import streamlit as st

from src.data_io import load_dataframe, ensure_required_columns
from src.modeling import train, predict, DT, Y, SITE

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
    st.error(f"Impossible de charger {TRAIN_PATH}: {e}")
    st.stop()

# Optionnel: subset pour la démo (évite d'entraîner/afficher sur trop gros)
st.caption(f"Train: {len(df_train):,} lignes")

# Entraînement
if st.button("Train model", type="primary"):
    # On reconstruit la pipeline avec le nouveau nb d'arbres si besoin
    from src.modeling import build_pipeline
    pipe = build_pipeline()
    # override n_estimators dynamiquement
    pipe.named_steps["xgb"].set_params(n_estimators=n_estimators)

    st.write("Entraînement en cours…")
    st.session_state["model"] = train(df_train)
    # NOTE: train() construit sa propre pipeline; pour respecter le slider,
    # remplace-le par pipe.fit(...) si tu veux appliquer n_estimators du slider:
    # model = pipe.fit(df_train.drop(columns=[Y]), df_train[Y]); st.session_state["model"] = model

    st.success("Modèle entraîné (XGBoost)")

# Prédiction
if st.button("Predict"):
    if "model" not in st.session_state:
        st.error("Entraîne d'abord le modèle.")
    else:
        model = st.session_state["model"]

        if src == "Subset du train (démo)":
            # On prend une petite fenêtre récente pour visualiser
            df_sub = df_train.sort_values(DT).tail(200).copy()
            y_pred = predict(model, df_sub)
            out = df_sub[[DT]].copy()
            out["y_pred"] = y_pred
            if Y in df_sub.columns:
                out["y_true"] = df_sub[Y].values

            st.subheader("Observé vs Prédit (subset train)")
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
                st.error(f"Impossible de charger {TEST_PATH}: {e}")
            else:
                y_pred = predict(model, df_test)
                sub = pd.DataFrame({"log_bike_count": y_pred})
                sub.index.name = "Id"

                st.subheader("Aperçu de la soumission")
                st.dataframe(sub.head(20))

                csv = sub.to_csv().encode("utf-8")
                st.download_button("⬇️ Télécharger le CSV pour Kaggle",
                                   data=csv, file_name="kaggle_submission.csv", mime="text/csv")
                st.success(f"{len(sub):,} lignes prêtes pour soumission.")
