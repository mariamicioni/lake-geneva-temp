# Lake Geneva Temperature Forecast

Predicts tomorrow's average water temperature of Lake Geneva at Geneva (BAFU station 2606, Rhône at Genève Halle de l'Île, where the lake drains) and compares the forecast against a persistence baseline ("tomorrow = today").

Built as the MLOps HS26 semester project (HSLU) using the Feature / Training / Inference (FTI) pipeline architecture.

## Data sources
- **Water:** FOEN/BAFU hydrology data (water temperature and flow, 10-min resolution) from the official [BAFU Data Platform](https://data.bafu.admin.ch), the same source as the training history (backfill: 10-min values and daily means back to 2016). Fallback when BAFU does not answer: the same readings via [api.existenz.ch](https://api.existenz.ch) (the main source until 27.10.2026).
- **Weather:** [Open-Meteo](https://open-meteo.com). The features use the Previous Runs API [6]: for every hour, the forecast made 48 hours earlier, which always existed before the 18:00 cutoff (complete from 20.01.2024). For a comparison in MS3, the daily job also saves the forecast of the ECMWF IFS run started at 06:00 UTC (Single Runs API [9]: published by about 12:00 UTC, so also before the cutoff; archived from 10.08.2024), and the newest forecast as an audit copy. All weather times are saved in UTC and converted to Swiss time with summer time in `features.py` (Open-Meteo's own "Europe/Zurich" uses one fixed offset for a whole download).

## Where the data lives
Data and models are never stored in Git; Git only holds code, configuration and small DVC pointer files. The only exception is a 4-day test sample of real BAFU readings in `tests/fixtures/` (22 KB), so that CI can test the feature code without the data.
- **Raw data and features (MS2):** versioned with DVC [5]. The files are in the private Google Cloud Storage bucket `gs://lake-geneva-temp-data` (folder `dvc/`); Git holds only the pointer files `data/raw.dvc` (all raw downloads) and `data/features.dvc` (the feature table `features.parquet`, built by `src/lakegeneva/features/build_features.py`). Data versions are Git tags: `data-v1` = the raw data until 27.10.2026 and the first feature table.
- **Get the data** (needs a key of the bucket's service account): `uv tool install "dvc[gs]"`, then `dvc remote modify --local storage credentialpath <path to the key>` and `dvc pull`. Without a key, the tests still run on the sample in `tests/fixtures/`.
- **Daily update:** the feature pipeline (`.github/workflows/feature-pipeline.yml`) runs every evening after the 18:00 cutoff: `dvc pull`, the new downloads, the whole feature table rebuilt, `dvc push`, and a commit of the two pointer files by `github-actions[bot]` ("chore(data): daily update ..."). So every day is one data version in Git, and every model can name the data it was trained on.
- **Before 27.10.2026** the daily snapshots went to the [`raw-data` release](https://github.com/mariamicioni/lake-geneva-temp/releases/tag/raw-data) as a backup; all of them are in `data-v1`, and the release is no longer updated.
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

## Where the code comes from
I build this project on what the MLOps HS26 course teaches. When a file follows a course exercise, its header names the source. This table grows with every milestone.

| File | Based on | What I changed |
|---|---|---|
| `.github/workflows/ci.yml` | Block 1 exercise, Part D and the image job of its solution [5] | No model training step yet; runs the unit tests, then builds the image without publishing it |
| `.github/workflows/feature-pipeline.yml` | Block 1 exercise, Parts A (DVC) and D (CI with uv) [5] | Runs on a schedule; pulls and pushes the data with DVC and commits the pointer files |
| `pyproject.toml`, `uv.lock` | uv dependency management [4] | Own package layout under `src/lakegeneva/` |
| `Dockerfile`, `.dockerignore` | Block 1 exercise, Part C (multi-stage image) [5] | Python 3.12 like the rest of the project, uv pinned to 0.13.0, our package copied to `/app/src`; no model, no HEALTHCHECK/EXPOSE (the feature pipeline is a job, not a web service); data mounted at run time |
| `src/lakegeneva/features/ingest_bafu.py` | BAFU Data Platform API documentation [1] | Written for this project |
| `src/lakegeneva/features/ingest_water.py` | existenz.ch Hydrology API documentation [2], BAFU Data Platform API documentation [1] (fallback and backfill) | Written for this project |
| `src/lakegeneva/features/ingest_weather.py` | Open-Meteo Forecast API documentation [3] | Written for this project |
| `src/lakegeneva/features/ingest_weather_runs.py` | Open-Meteo Previous Runs API documentation [6] | Written for this project |
| `src/lakegeneva/features/ingest_weather_single.py` | Open-Meteo Single Runs API documentation [9] | Written for this project |
| `src/lakegeneva/common/features.py` | The features of my proposal (`docs/proposal.pdf`), pandas documentation [7] | Written for this project |
| `src/lakegeneva/features/build_features.py` | pandas documentation [7] (Parquet) | Written for this project |
| `.dvc/config`, `data/raw.dvc`, `data/features.dvc` | Block 1 exercise, Part A (version the data with DVC) [5] | Google Cloud Storage instead of the local S3 bucket; the key path only in `.dvc/config.local` (not in Git) |
| `src/lakegeneva/common/baselines.py` | Trend-adjusted persistence and bootstrap interval suggested in the MS1 reviews (A1, A2/B2); moving-block bootstrap [8] | Written for this project |

No code is copied from past student projects (mlops-lab.ch, KTH ID2223).

**AI use:** Anthropic Claude (Claude Code) helped me write and test the code and the documentation. I reviewed every line and can explain it, and I am responsible for all technical decisions.

## References
[1] FOEN/BAFU (2026). BAFU Data Platform, Hydrological Observations API (GraphQL). accessed on 11.10.2026 from data.bafu.admin.ch
[2] Existenz.ch (2026). Hydrology API (republished BAFU data). accessed on 01.10.2026 from api.existenz.ch
[3] Open-Meteo (2026). Weather Forecast API. accessed on 20.10.2026 from open-meteo.com/en/docs
[4] Bär, Florian (2026). Versioning and Containerization, MLOps HS26 Block 1, lecture slides. Lucerne University of Applied Sciences and Arts.
[5] Bär, Florian (2026). Course project, Block 1: version the data, track the model, ship the service (exercise_quotes_1_version_and_ship), Part A: DVC, Part C: Docker, Part D: CI. MLOps HS26, Lucerne University of Applied Sciences and Arts.
[6] Open-Meteo (2026). Previous Runs API. accessed on 27.10.2026 from open-meteo.com/en/docs/previous-runs-api
[7] pandas development team (2026). pandas documentation, version 3.1. accessed on 27.10.2026 from pandas.pydata.org/docs
[8] Künsch, Hans R. (1989). The Jackknife and the Bootstrap for General Stationary Observations. The Annals of Statistics 17(3), 1217-1241.
[9] Open-Meteo (2026). Single Runs API. accessed on 27.10.2026 from open-meteo.com/en/docs/single-runs-api
