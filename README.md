# bike-count-streamlit

## Docker

Image on Docker Hub: https://hub.docker.com/r/leatinelli/bike-paris

# Paris Bike Counts — Streamlit App (Tests, CI & Docker)

![CI](https://github.com/leatinellix/bike-count-streamlit/actions/workflows/ci.yml/badge.svg)

This project turns a Kaggle-style bike counting task into a clean, reproducible product:
- An **interactive Streamlit app** to explore and **forecast** hourly bike counts in Paris.
- A **containerized** app (Docker) for identical runs anywhere.
- **Tests** for data importing / filtering / geo utilities.
- **CI** (GitHub Actions) that runs tests on every push/PR.

---

## Problem statement (what this project does)

We predict the **hourly number of bikes** passing at counting stations across Paris.  
- Each station is identified by a **`site_id`**.  
- The **training set** covers a **full year** of historical counts;  
- The **prediction set** covers the **following year** (same stations).  
- The target used for modeling is **`log_bike_count`** (log1p of the raw counts to stabilize variance).

**Model:** Gradient-boosted trees (**XGBoost**) inside an sklearn pipeline.  
**Hyperparameter search:** **Optuna**, integrated in the app (you can run trials from the UI).

---

## Data requirements

Put your files in `data/` (they are intentionally git-ignored):

data/train.parquet # required – the training year (hourly rows, multiple stations)
data/final_test.parquet # optional – the next year to predict (for CSV export + predictions map)


Minimum columns:
- `date` (timestamp, hourly)
- `log_bike_count` (`log1p(bike_count)`)
- `site_id` (string/int) 
- `site_name` 
- `coordinates` (optional: "48.8566,2.3522" or "POINT (2.3522 48.8566)")

**Feature engineering used in the pipeline**
- Calendar: `year`, `month`, `day`, `weekday`, `hour`
- Special days: French **holidays**, **school vacations**, and **lockdown/curfew** flags (COVID period)
- Optional external merge: hourly **weather** (if present)

---

## Project structure
app/
app.py # Overview: Map / Time series / Table
pages/02_Forecast.py # Forecast: cutoff, Optuna, Train, Predict (+ predictions map)
src/
data_io.py # loading, validation, timezone handling
filtering.py # reusable filters (date range, stations)
modeling.py # sklearn pipeline + XGBRegressor helpers
geo.py # lat/lon parsing, coords cache merge, geocoding glue
tests/
test_data_io.py # read + schema + tz robustness
test_filtering.py # date/station filters
test_geo.py # coordinates parsing/merging logic
scripts/
build_site_coords.py # optional: build external_data/site_coords.csv via geocoding
external_data/
site_coords.csv # optional cache (can be committed)
.github/workflows/
ci.yml # CI: pytest on push/PR
requirements.txt
Dockerfile
.dockerignore
README.md


---

## Quick start (local)

Requires **Python 3.12**.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. streamlit run app/app.py
# open http://localhost:8501

Overview page

Filters (left sidebar): date range, stations (site_id), aggregation (Hourly/Daily/Weekly).

Map of Paris: bubbles sized by total bike counts over the selected period.

Needs lat/lon in your data or a cache file external_data/site_coords.csv.

Time series: per-station lines, with a smoothing window and an optional average line.

Table: head of the filtered dataset.

Forecast page

Time split cutoff (sidebar):

Train = rows with date ≤ cutoff

Validation = rows with date > cutoff

(Optional) Run Optuna (choose n_trials).
Best params are displayed and reused automatically.

Train model.
Uses Optuna’s best params if available; otherwise uses the UI sliders.

Predict.

Shows Validation MAE (log and raw counts).

Training & validation data: Observed vs Predicted chart on a recent subset.

Test data: Kaggle-style CSV + Predictions map (bubbles sized by total predicted counts).

Notes
• Primary optimization metric = MAE on log_bike_count (stable).
• MAE on raw counts is also reported (interpretability).

## Tests

PYTHONPATH=. pytest -q

Covers: data loading/validation (including DST edge cases), filtering logic, and coordinates parsing/merging.

## Continuous Integration (CI)

GitHub Actions runs tests on every push/PR.
Workflow: .github/workflows/ci.yml
Badge (top of this README) reflects the latest run.

## Docker (containerized app)

1. Build docker image

docker build -t bike-paris:latest .

2. Run the container

docker run --rm -it -p 8501:8501 \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/external_data:/app/external_data:ro" \
  bike-paris:latest
# open http://localhost:8501

3. Push to docker hub

docker login

docker tag bike-paris:latest your-dockerhub-username/bike-paris:latest

docker push your-dockerhub-username/bike-paris:latest

docker pull your-dockerhub-username/bike-paris:latest  
docker run -p 8501:8501 your-dockerhub-username/bike-paris:latest

4. Or use the published image on Docker Hub :

docker pull leatinelli/bike-paris:latest
docker run --rm -it -p 8501:8501 \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/external_data:/app/external_data:ro" \
  leatinelli/bike-paris:latest

## Coordinates

If your data lacks coordinates, add:
external_data/site_coords.csv   # columns: site_id, site_name, lat, lon

You can generate it from station names:
pip install geopy
PYTHONPATH=. python scripts/build_site_coords.py

You may also edit lat/lon manually in the CSV.
This single file can be committed so the map works out-of-the-box.








