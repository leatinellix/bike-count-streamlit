# src/features.py
import pandas as pd
import numpy as np

FR_HOLIDAYS_2020_2021 = pd.to_datetime([
    "2020-11-01", "2020-11-11", "2020-12-25",
    "2021-01-01", "2021-04-05", "2021-05-01",
    "2021-05-08", "2021-05-13", "2021-05-24",
    "2021-07-14", "2021-08-15"
])

def encode_dates(X: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    X = X.copy()
    X[date_col] = pd.to_datetime(X[date_col], errors="coerce")

    X["year"] = X[date_col].dt.year
    X["month"] = X[date_col].dt.month
    X["day"] = X[date_col].dt.day
    X["weekday"] = X[date_col].dt.weekday
    X["hour"] = X[date_col].dt.hour

    # encodage cyclique (heures & mois)
    X["hour_sin"]  = np.sin(2*np.pi*X["hour"]/24.0)
    X["hour_cos"]  = np.cos(2*np.pi*X["hour"]/24.0)
    X["month_sin"] = np.sin(2*np.pi*X["month"]/12.0)
    X["month_cos"] = np.cos(2*np.pi*X["month"]/12.0)

    X["is_weekend"] = (X["weekday"] >= 5).astype(int)
    X["is_holiday"] = X[date_col].dt.normalize().isin(FR_HOLIDAYS_2020_2021).astype(int)
    return X

