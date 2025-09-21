# bike-count-streamlit

## Docker

Image on Docker Hub: https://hub.docker.com/r/leatinelli/bike-paris

# Paris Bike Counts — Streamlit App (Tests, CI & Docker)

![CI](https://github.com/leatinellix/bike-count-streamlit/actions/workflows/ci.yml/badge.svg)

Interactive Streamlit app to explore and **forecast** hourly bike counts in Paris.

- **Problem**: predict the **hourly number of bikes** per counting station (**`site_id`**).
- **Data**: one **training year** → predict the **following year**.
- **Target**: `log_bike_count` (log1p of raw counts) for stability.
- **Model**: **XGBoost** (sklearn pipeline) + **Optuna** for hyperparameter search.
- **Engineering**: tests (data import / filtering / geo), **CI** (GitHub Actions), **Docker** image.

---

## Features

- **Overview**
  - Filters (date range, stations)
  - **Map of Paris** (bubble = total counts over selected period)
  - **Time series** (smoothing + average line)
  - **Table** preview
- **Forecast**
  - Time cutoff (Train ≤ cutoff, Validation > cutoff)
  - **Run Optuna** → **Train** → **Predict**
  - Validation MAE (log & raw) + **Predictions map** on test
- **Quality**
  - Pytest (data IO / filtering / geo), CI on each push/PR, Dockerized app

---

## Data requirements

Put files in `data/` (not tracked by git):

data/train.parquet # required – hourly training year (many stations)
data/final_test.parquet # optional – next year to predict (CSV + predictions map)


Minimum columns:
- `date` (hourly timestamp)
- `log_bike_count` (`log1p(bike_count)`)
- `site_id`, `site_name`

**Map coordinates cache (optional)**  
`external_data/site_coords.csv` with columns: `site_id,site_name,lat,lon`.  
Build from station names:
~~~bash
pip install geopy
PYTHONPATH=. python scripts/build_site_coords.py
~~~

---

## Quick start (local)

Requires **Python 3.12**.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. streamlit run app/app.py
# open http://localhost:8501
~~~

Windows (PowerShell):
~~~powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="."
streamlit run app/app.py
~~~

---

## How to use the app

**Overview**
- Pick date range & stations (`site_id`) + aggregation (Hourly/Daily/Weekly)
- Map = bubble size by total counts (needs coords or CSV cache)
- Time series per station (smoothing + average)
- Table preview

**Forecast**
1. Set **time split cutoff**  
   - Train = rows with `date ≤ cutoff`  
   - Validation = rows with `date > cutoff`
2. *(Optional)* **Run Optuna** (`n_trials`) → best params reused automatically
3. **Train model** (XGBRegressor)
4. **Predict**
   - Validation **MAE** (log & raw)
   - **Train/valid chart** (Observed vs Predicted)
   - On **Test data**: Kaggle-style CSV + **Predictions map**

> Notes  
> • Primary metric = **MAE on `log_bike_count`** (stable)  
> • Raw-count MAE also reported (interpretability)

---

## Tests

~~~bash
PYTHONPATH=. pytest -q
~~~

Covers: data loading/validation (incl. DST edge cases), filtering logic, coordinates parsing/merge.

---

## Continuous Integration (CI)

GitHub Actions runs pytest on every push/PR.  
Workflow: `.github/workflows/ci.yml` — badge at top shows latest status.

---

## Docker

**Build locally**
~~~bash
docker build -t bike-paris:latest .
~~~

**Run (mount your local data)**
~~~bash
docker run --rm -it -p 8501:8501 \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/external_data:/app/external_data:ro" \
  bike-paris:latest
# open http://localhost:8501
~~~

**Or use the published image**
~~~bash
docker pull leatinelli/bike-paris:latest
docker run --rm -it -p 8501:8501 \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/external_data:/app/external_data:ro" \
  leatinelli/bike-paris:latest
~~~

---

## Project structure

app/ # Streamlit UI (Overview + Forecast)
src/ # data_io, filtering, modeling (XGB), geo
tests/ # pytest (data IO / filtering / geo)
scripts/ # build_site_coords.py (optional)
external_data/ # site_coords.csv (optional cache)
.github/workflows/ # ci.yml (pytest on push/PR)
requirements.txt
Dockerfile
.dockerignore
README.md










