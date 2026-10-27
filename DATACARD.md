# Data card: Lake Geneva Temperature Forecast

What is in the data, in which units, and what happens with gaps. (Recommended in the MLOps HS26 Block 1 lecture: "document column names, dtypes and missing-value handling in a DATACARD.md".) State of 27.10.2026.

## The feature table
- **File:** `data/features/features.parquet` (Parquet, versioned with DVC), built by `src/lakegeneva/features/build_features.py` with the shared definition in `src/lakegeneva/common/features.py`.
- **One row = one issue day D**, a Swiss calendar day (Europe/Zurich, with summer time). The forecast is made at 18:00 on D for the mean of D+1.
- **Index `date`:** the day D (from 31.12.2015). 3,954 rows on 27.10.2026, one more every day; 1,009 rows have all features and a label (from 21.01.2024).

| Column | Unit | Meaning | Known at 18:00 on D? | Empty when |
|---|---|---|---|---|
| `n_readings_d0` | count | water temperature readings of D before 18:00 (108 on a normal day, 102 or 114 on a clock-change day) | yes | never (0 if there is none) |
| `water_mean_d0` | °C | mean water temperature of D from 00:00 to 18:00 | yes | fewer than 90 of the 108 readings before 18:00 |
| `water_last` | °C | the last reading before 18:00 (also the persistence baseline) | yes | as above |
| `water_mean_3d` | °C | mean of all readings of D-2 and D-1 (whole days) and of D until 18:00 | yes | as above |
| `water_trend_3d` | °C | `water_mean_d0` of D minus `water_mean_d0` of D-3 (the same hours on both days) | yes | the rule above fails on D or D-3 |
| `flow_mean_d0` | m³/s | mean discharge of D until 18:00 | yes | as for `water_mean_d0` |
| `air_temp_fc_d1` | °C | mean air temperature of D+1, from the 48-hour-old forecast | yes (made at least 48 h before) | fewer than 20 complete forecast hours for D+1 |
| `sunshine_fc_d1` | h | hours of sunshine of D+1 (hourly seconds summed, / 3600) | yes | as above |
| `wind_max_fc_d1` | km/h | strongest hourly wind of D+1 | yes | as above |
| `wind_sw_fc_d0` | km/h | mean south-west part of the wind of D: speed × cos(direction − 225°) per hour, then the mean; positive for wind from the south-west, negative for the bise (north-east) | yes (a forecast, not a measurement) | fewer than 20 complete forecast hours for D |
| `wind_sw_fc_d1` | km/h | the same for D+1 | yes | fewer than 20 complete forecast hours for D+1 |
| `n_readings_d1` | count | readings of D+1 (144, or 138/150 on a clock-change day) | no (belongs to the label) | never (0 while D+1 has not started) |
| `label_water_mean_d1` | °C | **the label**: mean of all readings of D+1 | **no**: measured tomorrow, never a feature | fewer than 120 readings of D+1, or D+1 not over yet |

## Raw files (`data/raw/`, versioned with DVC)
| File | Source | One row per | Time column | Written by |
|---|---|---|---|---|
| `water_2606_10min_<start>_to_<end>.csv` | BAFU Data Platform, 10-minute means of water temperature (WT) and discharge (Q) | 10 minutes | `time_utc` | `ingest_water.py --start` (backfill) |
| `water_2606_fetched_<date>.csv` | the last 7 days: BAFU from 27.10.2026, before that existenz.ch | 10 minutes | `time_utc` | `ingest_water.py` (daily) |
| `water_2606_backfill_2026-08-28_to_2026-09-29.csv` | existenz.ch, the first 32-day download (29.09.2026) | 10 minutes | `time_utc` | an early version of `ingest_water.py` |
| `bafu_2606_daily_<start>_to_<end>.csv` | BAFU daily means (days in UTC+1 all year) | day | `date` | `ingest_bafu.py` (context only, not used by the features) |
| `weather_geneva_prev2_<start>_to_<end>.csv` | Open-Meteo Previous Runs: for every hour the forecast made 48 hours earlier (`*_previous_day2`) | hour | `time_utc` (older files: `time_local`, see "Time") | `ingest_weather_runs.py` |
| `weather_geneva_single06_<start>_to_<end>.csv` | Open-Meteo Single Runs: the ECMWF IFS run started at 06:00 UTC on each day, 48 hours | run and hour | `run_utc`, `time_utc` | `ingest_weather_single.py` (for a comparison in MS3, not a feature yet) |
| `weather_geneva_fetched_<date>.csv` | Open-Meteo Forecast API: the newest forecast at the download | hour (72) | `time_utc` (older files: `time_local`) | `ingest_weather.py` (audit copy, never a feature: downloaded after 18:00) |
| `bafu_2606_2026-09-29.csv`, `.json` | the first test download (29.09.2026) | | | not used |

Units in the raw files: `water_temp_c` °C, `flow_m3s` m³/s, `temperature_2m` °C, `sunshine_duration` seconds of sun in the hour (0-3600), `wind_speed_10m` km/h, `wind_direction_10m` degrees the wind comes from (0 = north, 90 = east, 225 = south-west). A missing value is an empty cell, never 0.

## Time
- **All raw times are UTC** (`time_utc`); `features.py` converts them to Europe/Zurich with summer time.
- **Old weather files** (downloaded before 27.10.2026) have `time_local`: Open-Meteo had shifted all hours by the UTC offset that was valid at the download (+1 h in winter, also for summer days). `load_weather()` undoes this with the download time `fetched_at_utc`.
- **The 18:00 cutoff** is applied to the time stamp of each reading, not to the time the job runs: a late run gives the same table.

## Gaps and rules
- **Water history (BAFU, 2016-2026):** no day below 120 readings (checked on 27.10.2026); BAFU's history is checked and complete.
- **Live data:** existenz.ch had days with only 10-12 readings (five Sundays in August/September 2026). Hence the rules: the label needs at least 120 of 144 readings, the features at least 90 of the 108 readings before 18:00, a forecast day at least 20 of its 24 hours (all 83 %).
- **The same reading in several files:** water: the file later in name order wins (the newest daily download); weather: the newest download wins. BAFU and existenz.ch differ by less than 0.01 °C on average.
- **BAFU's newest 10-minute values are about 1.5 hours old** (measured on 27.10.2026), so the daily job runs at 19:00 UTC, well after the cutoff.
- **Weather history:** the 48-hour-old sunshine and wind exist from 20.01.2024 13:00 (air temperature from 2022). The 06:00 UTC single runs exist from 10.08.2024; one run is missing in the archive (05.08.2025); the first hour of every run has no sunshine value.
- **Clock-change days** have 23 or 25 hours (138 or 150 readings) and count as complete.
- **Extreme but real values:** `water_trend_3d` goes from -11.5 °C (upwelling after south-west wind, 5-8 July 2025) to +9.6 °C (June 2024).

## Versions
- One Git commit of `data/raw.dvc` and `data/features.dvc` = one data version. The daily job commits them as `github-actions[bot]`; `data-v1` (27.10.2026) is the first tag.
- Get an older version: `git checkout <commit or tag>`, then `dvc checkout` (or `dvc pull`).
