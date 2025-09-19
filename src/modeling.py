from typing import Optional
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from xgboost import XGBRegressor
from .features import encode_dates

DT, Y, SITE = "date", "log_bike_count", "site_id"

def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = encode_dates(df, date_col=DT)
    # on garde seulement les colonnes utiles au modèle
    out = df[[ "year","month","day","weekday","hour", SITE ]].copy()
    out[SITE] = out[SITE].astype(str)  # cat
    return out

def build_pipeline() -> Pipeline:
    cat_cols = ["year","month","day","weekday","hour", SITE]
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols)],
        remainder="drop",
    )
    model = XGBRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42,
        objective="reg:squarederror", n_jobs=4
    )
    pipe = Pipeline([
        ("prep", FunctionTransformer(_make_features, validate=False)),
        ("pre", pre),
        ("xgb", model),
    ])
    return pipe

def train(df_train: pd.DataFrame) -> Pipeline:
    X = df_train.drop(columns=[Y], errors="ignore")
    y = df_train[Y]
    pipe = build_pipeline()
    pipe.fit(X, y)
    return pipe

def predict(model: Pipeline, df_new: pd.DataFrame):
    return model.predict(df_new)
