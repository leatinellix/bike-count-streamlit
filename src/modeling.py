# src/modeling.py
from typing import Optional, Dict
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from xgboost import XGBRegressor
from .features import encode_dates

# Colonnes canon
DT, Y, SITE = "date", "log_bike_count", "site_id"

def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les features attendues par le modèle à partir de df.
    Requiert que encode_dates ait créé: weekday, hour, hour_sin/cos, month_sin/cos, is_weekend, is_holiday.
    """
    X = encode_dates(df, date_col=DT).copy()
    if SITE not in X.columns:
        X[SITE] = "none"
    X[SITE] = X[SITE].astype(str)

    cat_cols = ["weekday", "hour", SITE]
    num_cols = ["hour_sin","hour_cos","month_sin","month_cos","is_weekend","is_holiday"]

    # S'assure que toutes les colonnes existent
    for c in cat_cols + num_cols:
        if c not in X.columns:
            X[c] = 0

    return X[cat_cols + num_cols]

def build_pipeline(xgb_params: Optional[Dict] = None) -> Pipeline:
    """Pipeline (features -> OneHot + num -> XGB)."""
    cat_cols = ["weekday", "hour", SITE]
    num_cols = ["hour_sin","hour_cos","month_sin","month_cos","is_weekend","is_holiday"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    default = dict(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, reg_alpha=0.0,
        objective="reg:squarederror", random_state=42, n_jobs=4,
        tree_method="hist",
    )
    if xgb_params:
        default.update(xgb_params)

    model = XGBRegressor(**default)
    return Pipeline([
        ("prep", FunctionTransformer(_make_features, validate=False)),
        ("pre", pre),
        ("xgb", model),
    ])

def train(df_train: pd.DataFrame, xgb_params: Optional[Dict] = None) -> Pipeline:
    """
    Entraîne le pipeline sur df_train. Si Y manque mais 'bike_count' existe,
    on utilise log1p(bike_count) comme cible.
    """
    X = df_train.copy()
    if Y not in X.columns and "bike_count" in X.columns:
        X[Y] = np.log1p(X["bike_count"])
    if Y not in X.columns:
        raise ValueError(f"Colonne cible manquante: {Y}")

    y = X[Y].to_numpy()
    X = X.drop(columns=[Y], errors="ignore")

    pipe = build_pipeline(xgb_params)
    pipe.fit(X, y)
    return pipe

def predict(model: Pipeline, df_new: pd.DataFrame) -> np.ndarray:
    """Prédit log_bike_count pour df_new."""
    return model.predict(df_new)

