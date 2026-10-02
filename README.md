# Lake Geneva Temperature Forecast

Predicts tomorrow's average water temperature of Lake Geneva at Geneva (BAFU station 2606, Rhône at Genève Halle de l'Île, where the lake drains) and compares the forecast against a persistence baseline ("tomorrow = today").

Built as the MLOps HS26 semester project (HSLU) using the Feature / Training / Inference (FTI) pipeline architecture.

## Data sources
- **Water:** FOEN/BAFU hydrology data via [api.existenz.ch](https://api.existenz.ch) (water temperature and flow, 10-min resolution). Fallback: official BAFU feed via LINDAS (ld.admin.ch).
- **Weather:** [Open-Meteo](https://open-meteo.com) forecast API; backfill from the Single Runs API (forecasts published before the 18:00 cutoff).

## Where the data lives
Data and models are never stored in Git; Git only holds code, configuration and small DVC pointer files.
- **Now (MS1):** a daily GitHub Actions job (`.github/workflows/collect-water.yml`) saves raw CSV snapshots as a temporary backup in the [`raw-data` release](https://github.com/mariamicioni/lake-geneva-temp/releases/tag/raw-data), which is outside the Git history.
- **From MS2:** the job writes raw data and features to Google Cloud Storage, versioned with DVC; the GitHub Release is then removed.
- **Models:** MLflow tracking and model registry hosted on DagsHub (alias `champion` = model in production).

## Status
- [ ] MS1 – Proposal (`docs/proposal.pdf`)
- [ ] MS2 – Feature pipeline
- [ ] MS3 – Training pipeline
- [ ] MS4 – Live system

## Repository layout
```
src/lakegeneva/
  features/    feature pipeline (ingest, backfill, compute, write)
  training/    training pipeline (read features, train, evaluate, register)
  inference/   inference pipeline (load model, predict, serve)
  common/      shared config and feature definitions
tests/         unit tests (run in CI)
config/        settings without secrets
notebooks/     exploration only
docs/          proposal and milestone summaries
```
