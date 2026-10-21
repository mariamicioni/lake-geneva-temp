# Lake Geneva Temperature Forecast

Predicts tomorrow's average water temperature of Lake Geneva at Geneva (BAFU station 2606, Rhône at Genève Halle de l'Île, where the lake drains) and compares the forecast against a persistence baseline ("tomorrow = today").

Built as the MLOps HS26 semester project (HSLU) using the Feature / Training / Inference (FTI) pipeline architecture.

## Data sources
- **Water:** FOEN/BAFU hydrology data via [api.existenz.ch](https://api.existenz.ch) (water temperature and flow, 10-min resolution). Fallback when existenz.ch does not answer: the same 10-min values from the official [BAFU Data Platform](https://data.bafu.admin.ch).
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

## Where the code comes from
I build this project on what the MLOps HS26 course teaches. When a file follows a course exercise, its header names the source. This table grows with every milestone.

| File | Based on | What I changed |
|---|---|---|
| `.github/workflows/ci.yml` | Block 1 exercise, Part D [5] | No model training step yet; runs the unit tests only |
| `pyproject.toml`, `uv.lock` | uv dependency management [4] | Own package layout under `src/lakegeneva/` |
| `src/lakegeneva/features/ingest_bafu.py` | BAFU Data Platform API documentation [1] | Written for this project |
| `src/lakegeneva/features/ingest_water.py` | existenz.ch Hydrology API documentation [2], BAFU Data Platform API documentation [1] (fallback) | Written for this project |
| `src/lakegeneva/features/ingest_weather.py` | Open-Meteo Forecast API documentation [3] | Written for this project |

No code is copied from past student projects (mlops-lab.ch, KTH ID2223).

**AI use:** Anthropic Claude (Claude Code) helped me write and test the code and the documentation. I reviewed every line and can explain it, and I am responsible for all technical decisions.

## References
[1] FOEN/BAFU (2026). BAFU Data Platform, Hydrological Observations API (GraphQL). accessed on 11.10.2026 from data.bafu.admin.ch
[2] Existenz.ch (2026). Hydrology API (republished BAFU data). accessed on 01.10.2026 from api.existenz.ch
[3] Open-Meteo (2026). Weather Forecast API. accessed on 20.10.2026 from open-meteo.com/en/docs
[4] Bär, Florian (2026). Versioning and Containerization, MLOps HS26 Block 1, lecture slides. Lucerne University of Applied Sciences and Arts.
[5] Bär, Florian (2026). Course project, Block 1: version the data, track the model, ship the service (exercise_quotes_1_version_and_ship), Part D: CI. MLOps HS26, Lucerne University of Applied Sciences and Arts.
