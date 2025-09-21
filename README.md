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




