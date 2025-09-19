import pandas as pd
from src.data_io import load_dataframe
from src.filtering import filter_by_date_range, filter_by_site, apply_filters

FIX = "tests/data_fixtures/sample.csv"
def _df(): return load_dataframe(FIX, datetime_col="date")

def test_date_range_filters_rows():
    df = _df()
    out = filter_by_date_range(df,
                               start=pd.Timestamp("2021-01-01 09:00:00"),
                               end=pd.Timestamp("2021-01-01 23:59:59"),
                               datetime_col="date")
    assert len(out) == 2  # 09:00 et 10:00 du 01/01

def test_site_filter_keeps_subset():
    df = _df()
    out = filter_by_site(df, sites=[75003], station_col="site_id")
    assert out["site_id"].nunique() == 1
    assert set(out["site_id"]) == {75003}

def test_apply_filters_combo():
    df = _df()
    out = apply_filters(df,
                        start=pd.Timestamp("2021-01-01 00:00:00"),
                        end=pd.Timestamp("2021-01-01 23:59:59"),
                        sites=[75001],
                        datetime_col="date",
                        station_col="site_id")
    assert len(out) == 2  # 01/01 pour site 75001
