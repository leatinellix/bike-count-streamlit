import os
from typing import List
import pandas as pd

def load_dataframe(path: str, datetime_col: str = "date") -> pd.DataFrame:
    """Charge un DataFrame depuis .parquet ou .csv (parse la colonne date si CSV)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif ext in {".csv", ".txt"}:
        df = pd.read_csv(path, parse_dates=[datetime_col])
    else:
        raise ValueError(f"Extension non supportée: {ext}")
    return df

def ensure_required_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Vérifie que les colonnes requises existent, sinon lève une erreur claire."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")
    return df

def ensure_datetime_tz(df: pd.DataFrame, datetime_col="date", tz="Europe/Paris") -> pd.DataFrame:
    """
    Force la colonne date en datetime + timezone.
    Gère les passages heure d'été/hiver :
      - ambiguous (ex: 2020-10-25 02:00 qui existe 2 fois) -> infer
      - nonexistent (printemps) -> shift_forward (décale d'1h)
    """
    out = df.copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce")

    # Si naïf -> on localise
    if out[datetime_col].dt.tz is None:
        try:
            out[datetime_col] = out[datetime_col].dt.tz_localize(
                tz, ambiguous="infer", nonexistent="shift_forward"
            )
        except TypeError:
            # fallback pour anciennes versions de pandas sans 'nonexistent'
            out[datetime_col] = out[datetime_col].dt.tz_localize(
                tz, ambiguous="infer"
            )
    else:
        # Si timezone déjà présente -> on convertit
        out[datetime_col] = out[datetime_col].dt.tz_convert(tz)

    return out


