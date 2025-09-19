import pandas as pd
import pytest
from src.data_io import load_dataframe, ensure_required_columns, ensure_datetime_tz

FIX = "tests/data_fixtures/sample.csv"

def test_load_dataframe_parses_dates():
    df = load_dataframe(FIX, datetime_col="date")
    assert pd.api.types.is_datetime64_any_dtype(df["date"])

def test_required_columns_ok():
    df = load_dataframe(FIX, datetime_col="date")
    ensure_required_columns(df, ["date", "log_bike_count", "site_id"])

def test_datetime_tz_paris():
    df = load_dataframe(FIX, datetime_col="date")
    out = ensure_datetime_tz(df, "date", "Europe/Paris")
    assert str(out["date"].dt.tz) == "Europe/Paris"

def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_dataframe("does_not_exist.csv")
