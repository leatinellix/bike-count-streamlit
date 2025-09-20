import os, sys, time
import pandas as pd

# --- ensure the repo root is on sys.path so "src" is importable ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.geo import geocode_site_names, make_nominatim_geocode_fn, save_site_coords

TRAIN_PATH = "data/train.parquet"
OUT_CSV = "external_data/site_coords.csv"

def main():
    # load unique stations
    df = pd.read_parquet(TRAIN_PATH)
    cols = [c for c in ["site_id", "site_name"] if c in df.columns]
    if not cols:
        raise SystemExit("Need 'site_id' or 'site_name' in train.parquet")
    stations = df[cols].drop_duplicates().sort_values(cols[0]).reset_index(drop=True)

    # real geocoder (Nominatim via geopy)
    geocode_fn = make_nominatim_geocode_fn()

    # polite rate limiting helper
    def throttled(q):
        res = geocode_fn(q)
        time.sleep(1.0)  # be nice to OSM
        return res

    coords = geocode_site_names(stations, geocode_fn=throttled)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    save_site_coords(coords, OUT_CSV)
    print(f"Saved {OUT_CSV} with {len(coords)} rows")

if __name__ == "__main__":
    main()
