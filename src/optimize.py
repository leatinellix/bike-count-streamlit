# src/optimize.py
from __future__ import annotations
import numpy as np, pandas as pd, optuna
from typing import Dict, Tuple
from sklearn.metrics import mean_absolute_error
from .modeling import build_pipeline, DT, Y

def _ensure_target(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    if Y not in X.columns and "bike_count" in X.columns:
        X[Y] = np.log1p(X["bike_count"])
    if Y not in X.columns:
        raise ValueError(f"Colonne cible manquante: {Y}")
    return X

def optimize_xgb(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    n_trials: int = 30,
    direction: str = "minimize",
) -> Tuple[Dict, float]:
    """
    Split temporel: train <= cutoff, valid > cutoff.
    Objectif: MAE sur l'échelle LOG (cohérent avec la cible).
    Retourne (meilleurs_params, meilleure_MAE_log).
    """
    df = _ensure_target(df)
    df = df.sort_values(DT)
    train_df = df[df[DT] <= cutoff]
    valid_df = df[df[DT] >  cutoff]

    X_train = train_df.drop(columns=[Y])
    y_train = train_df[Y]
    X_valid = valid_df.drop(columns=[Y])
    y_valid = valid_df[Y]

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators = trial.suggest_int("n_estimators", 300, 1200),
            learning_rate = trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
            max_depth = trial.suggest_int("max_depth", 4, 10),
            subsample = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda = trial.suggest_float("reg_lambda", 0.0, 5.0),
            reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0),
        )
        pipe = build_pipeline(params)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_valid)
        mae_log = mean_absolute_error(y_valid, y_pred)
        # pruning optionnel:
        trial.report(mae_log, step=1)
        return mae_log

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_trial.params
    best_score = float(study.best_value)
    return best_params, best_score
