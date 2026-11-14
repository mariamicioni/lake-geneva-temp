"""Training experiments: is the model worth it, how much data, how often to retrain?

Three questions from the SW09 lecture ("Your pipeline") and one from the MS2 summary,
answered on the test days from 21.01.2025, always with the walk-forward test of train.py:

1. Ladder: from the cheapest forecast to the full model, with error and training time.
2. Data: train only on the last 4, 6 or 12 months instead of all earlier rows.
3. Retrain: a new model every day, every week, every month or only once.
4. Forecast: the 48-hour-old forecast (the features now) against the forecast of the
   06:00 UTC model run of the same day (a promise of the MS2 summary).

    python src/lakegeneva/training/experiments.py
"""
# Based on the MLOps HS26 SW09 lecture (Efficiency Techniques), "Your pipeline"
# questions 1, 4 and 5, Dr. M. Bravin [14] in README.md. The lecture's own scripts
# are not used: they test a LightGBM model on hourly Lucerne temperatures.
# Why the same alpha in every experiment: it is chosen once on the validation year
# (as in train.py), so each experiment changes only one thing. With fewer training
# rows the same alpha is a slightly stronger penalty; that is part of the answer.

import argparse  # reads the command-line option --features
import time  # perf_counter() measures how long training takes

import numpy as np  # median of the measured times
import pandas as pd  # the daily feature table

from lakegeneva.common import baselines, features  # the baselines and column names
from lakegeneva.training import train  # the model and the walk-forward test

# The training windows of question 2, in days (None: all earlier rows). Not shorter
# than 4 months: a model needs at least train.MIN_TRAIN_ROWS = 90 rows.
WINDOWS = {"last 4 months": 120, "last 6 months": 182, "last 12 months": 365, "all earlier rows": None}
# How often a new model is trained in question 3, in days. 100,000 days = never again
# after the first test day.
RETRAIN = {"every day": 1, "every week": 7, "every month": 30, "only once": 100_000}
# How many times one fit is timed; the median ignores a single slow run.
TIMING_REPEATS = 5


def scores(rows: pd.DataFrame, forecast: pd.Series) -> dict:
    """MAE on all test days and on the big-change days (°C)."""
    big = baselines.big_change_days(rows.loc[forecast.index])
    return {
        "test MAE": train.mae(rows, forecast),
        "change > 0.8 MAE": train.mae(rows, forecast[big.to_numpy()]),
    }


def fit_seconds(rows: pd.DataFrame, columns: list[str], alpha: float) -> float:
    """Median time of one training fit on all rows (seconds)."""
    times = []
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        train.train_final(rows, columns, alpha)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def average_change(rows: pd.DataFrame, days) -> pd.Series:
    """Persistence plus the average change so far: no feature and no model."""
    # Why this rung: the last reading before 18:00 is an afternoon value, and on
    # average it is warmer than the mean of the next day (by 0.31 degC on 14.11.2026,
    # by 0.44 degC from April to September). Adding the average change of the rows
    # known at 18:00 corrects this with one number. So the Ridge rungs must beat THIS
    # rung to show that the features help; beating plain persistence is not enough.
    forecasts = {
        day: rows.at[day, "water_last"] + train.change(train.known_at_18(rows, day)).mean() for day in days
    }
    return pd.Series(forecasts, name="forecast", dtype=float).rename_axis("date")


def ladder(rows: pd.DataFrame, test_days, alphas: dict) -> pd.DataFrame:
    """Question 1: each rung, from free to the full model, on the same test days."""
    part = rows.loc[test_days]
    result = {
        # The first three rungs cost nothing: no training at all.
        "1 persistence": scores(rows, baselines.persistence(part)) | {"one fit (ms)": 0.0},
        "2 trend-adjusted persistence": scores(rows, baselines.trend_adjusted(part)) | {"one fit (ms)": 0.0},
        "3 persistence + average change": scores(rows, average_change(rows, test_days)) | {"one fit (ms)": 0.0},
    }
    for rung, name in (("4 Ridge, water only", "water_only"), ("5 Ridge, all features", "full")):
        columns = train.FEATURE_SETS[name]
        start = time.perf_counter()
        forecast = train.walk_forward(rows, columns, alphas[name], test_days)
        seconds = time.perf_counter() - start
        result[rung] = scores(rows, forecast) | {
            "one fit (ms)": 1000 * fit_seconds(rows, columns, alphas[name]),
            # One model per test day: the whole walk-forward test.
            "walk-forward (s)": seconds,
        }
    return pd.DataFrame(result).T


def learning_curve(rows: pd.DataFrame, test_days, alpha: float) -> pd.DataFrame:
    """Question 2: the full model trained on less data, on the same test days."""
    result = {}
    for name, window in WINDOWS.items():
        forecast = train.walk_forward(rows, train.FEATURES_FULL, alpha, test_days, window_days=window)
        result[name] = scores(rows, forecast) | {"days with a forecast": len(forecast)}
    return pd.DataFrame(result).T


def retrain_replay(rows: pd.DataFrame, test_days, alpha: float) -> pd.DataFrame:
    """Question 3: the full model retrained less often, on the same test days."""
    result = {}
    for name, every in RETRAIN.items():
        forecast = train.walk_forward(rows, train.FEATURES_FULL, alpha, test_days, retrain_every=every)
        result[name] = scores(rows, forecast)
    return pd.DataFrame(result).T


def forecast_comparison(rows: pd.DataFrame, single: pd.DataFrame) -> pd.DataFrame:
    """Question 4: the same model with the weather features from the 06:00 UTC run instead."""
    # The same water features and labels; only the five weather columns are replaced.
    with_single = rows.copy()
    with_single[features.WEATHER_FEATURES] = single.reindex(rows.index)[features.WEATHER_FEATURES]
    # A fair comparison: only the days on which both forecasts exist (the 06:00 UTC
    # runs start on 10.08.2024), for training and for testing.
    both = with_single[features.WEATHER_FEATURES].notna().all(axis=1)
    variants = {
        "48-hour-old forecast (the features now)": rows[both],
        "06:00 UTC run of the same day": with_single[both],
    }
    result = {}
    for name, part in variants.items():
        test_days = part.index[part.index >= baselines.TEST_START]
        # Each variant chooses its own setting on its validation days, as in train.py.
        alpha = train.choose_alpha(part, train.FEATURES_FULL)[0]
        forecast = train.walk_forward(part, train.FEATURES_FULL, alpha, test_days)
        result[name] = scores(part, forecast) | {"alpha": alpha, "test days": len(forecast)}
    return pd.DataFrame(result).T


def main() -> None:
    # Command-line options; __doc__ (the text at the top) becomes the --help text.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--features", default="data/features/features.parquet", help="the feature table (the feature store)"
    )
    # Question 4 needs the raw files of the 06:00 UTC runs: they are not in the feature
    # store, because the model does not use them (yet).
    parser.add_argument("--raw", default="data/raw", help="folder with the raw files")
    args = parser.parse_args()

    rows = train.model_rows(pd.read_parquet(args.features))
    test_days = rows.index[rows.index >= baselines.TEST_START]
    # The setting of each model, chosen on the validation year only (as in train.py).
    alphas = {name: train.choose_alpha(rows, columns)[0] for name, columns in train.FEATURE_SETS.items()}
    print(f"{len(test_days)} test days from {test_days.min():%d.%m.%Y}; alpha {alphas}; MAE in degC")

    print("\n1 Ladder: is the model worth its training?")
    print(ladder(rows, test_days, alphas).round(3).to_string())
    print("\n2 Data: the full model trained on less data")
    print(learning_curve(rows, test_days, alphas["full"]).round(3).to_string())
    print("\n3 Retrain: the full model retrained less often")
    print(retrain_replay(rows, test_days, alphas["full"]).round(3).to_string())
    print("\n4 Forecast: the 48-hour-old forecast against the 06:00 UTC run of the same day")
    single = features.single_run_weather_features(features.load_single_runs(args.raw))
    print(forecast_comparison(rows, single).round(3).to_string())


# Runs main() only when the file is started as a script, not when it is imported
# (the tests import this file).
if __name__ == "__main__":
    main()
