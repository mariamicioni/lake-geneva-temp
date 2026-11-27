"""Training experiments: is the model worth it, how much data, how often to retrain?

Three questions from the SW09 lecture ("Your pipeline") and two from the MS2 summary
and its review, answered on the test days from 21.01.2025, always with the walk-forward
test of train.py:

1. Ladder: from the cheapest forecast to the full model, with error and training time.
2. Data: train only on the last 4, 6 or 12 months instead of all earlier rows.
3. Retrain: a new model every day, every week, every month or only once.
4. Forecast: the 48-hour-old forecast (the features now) against the forecast of the
   06:00 UTC model run of the same day (a promise of the MS2 summary).
5. Weather point: Geneva against a point on the open lake (only with --lake-point,
   it downloads the lake's forecast; MS2 review point B2).

Every difference that the README names comes with its 95 % interval, from the same
bootstrap as the baselines, so every number of the README can be repeated.

    python src/lakegeneva/training/experiments.py
"""
# Based on the MLOps HS26 SW09 lecture (Efficiency Techniques), "Your pipeline"
# questions 1, 4 and 5, Dr. M. Bravin [14] in README.md. The lecture's own scripts
# are not used: they test a LightGBM model on hourly Lucerne temperatures.
# Why the same alpha in every experiment: it is chosen once on the validation year
# (as in train.py), so each experiment changes only one thing. With fewer training
# rows the same alpha is a slightly stronger penalty; that is part of the answer.

import argparse  # reads the command-line option --features
import datetime as dt  # type hint: the first and last day of the lake download
import time  # perf_counter() measures how long training takes
from collections.abc import Iterable  # type hint: anything a for loop can go through (a list, an index of dates)

import numpy as np  # median of the measured times; np.ones for the "all test days" window
import pandas as pd  # the daily feature table

from lakegeneva.common import baselines, feature_store, features, location  # the baselines and column names; the read path; the lake point
from lakegeneva.features import ingest_weather_runs  # the 48-hour-old forecast, here at the lake point (question 5)
from lakegeneva.training import train  # the model and the walk-forward test

# The training windows of question 2, in days (None: all earlier rows). Not shorter
# than 4 months: a model needs at least train.MIN_TRAIN_ROWS = 90 rows.
WINDOWS = {"last 4 months": 120, "last 6 months": 182, "last 12 months": 365, "all earlier rows": None}
# How often a new model is trained in question 3, in days. 100,000 days = never again
# after the first test day.
RETRAIN = {"every day": 1, "every week": 7, "every month": 30, "only once": 100_000}
# How many times one fit is timed; the median ignores a single slow run.
TIMING_REPEATS = 5


def scores(rows: pd.DataFrame, forecast: pd.Series, days: Iterable[pd.Timestamp]) -> dict:
    """MAE on all test days and on the big-change days (°C). Every day in `days` needs a forecast."""
    # The same rule as train.evaluate(): a missing forecast (NaN, or a day that is not
    # there) must stop the experiment instead of quietly dropping out of the mean. This
    # script makes the tables of the README (found in a review on 20.11.2026).
    train.check_every_day_forecast(forecast, days)
    big = baselines.big_change_days(rows.loc[forecast.index])
    return {
        "test MAE": train.mae(rows, forecast),
        "change > 0.8 MAE": train.mae(rows, forecast[big.to_numpy()]),
    }


def windows_of(part: pd.DataFrame) -> dict:
    """The windows of the README as yes/no lists over the days of `part`."""
    return {
        "all test days": np.ones(len(part), dtype=bool),
        "change > 0.8": baselines.big_change_days(part).to_numpy(),
        # October to January by the day that is forecast (tomorrow), as in train.py.
        "Oct-Jan": (part.index + pd.Timedelta(days=1)).month.isin([10, 11, 12, 1]),
    }


def difference(error_a: pd.Series, error_b: pd.Series, in_window: np.ndarray) -> dict:
    """MAE(a) - MAE(b) on the days in `in_window`: the gain of a in % and the 95 % interval."""
    # Below 0 means that a is better. The interval says how sure that is: if it holds
    # 0, the days cannot tell which one is better.
    row = {"days": int(in_window.sum())}
    if row["days"] > 0:
        mae_a, mae_b = error_a[in_window].mean(), error_b[in_window].mean()
        row |= {"diff": mae_a - mae_b, "gain_%": 100 * (1 - mae_a / mae_b)}
    # The same rule as in train.evaluate(): with fewer than 35 days an interval would
    # look far too sure, so there is none.
    if row["days"] >= baselines.MIN_DAYS_BOOTSTRAP:
        _, row["ci_low"], row["ci_high"] = baselines.bootstrap_mae_difference(
            error_a[in_window], error_b[in_window]
        )
    return row


def window_differences(error_a: pd.Series, error_b: pd.Series, part: pd.DataFrame) -> dict:
    """difference() in each window of the README; the big-change window also counts its events."""
    result = {}
    for window, in_window in windows_of(part).items():
        row = difference(error_a, error_b, in_window)
        if window == "change > 0.8":
            # Next to the day count: how many separate events these days are, as runs
            # of consecutive days (MS2 review point B3; train.py logs the same number).
            row = {"days": row.pop("days"), "events": baselines.count_events(part.index[in_window])} | row
        result[window] = row
    return result


def fit_seconds(rows: pd.DataFrame, columns: list[str], alpha: float) -> float:
    """Median time of one training fit on all rows (seconds)."""
    times = []
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        train.train_final(rows, columns, alpha)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def average_change(rows: pd.DataFrame, days: Iterable[pd.Timestamp]) -> pd.Series:
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


def ladder(rows: pd.DataFrame, test_days: pd.DatetimeIndex, alphas: dict) -> pd.DataFrame:
    """Question 1: each rung, from free to the full model, on the same test days."""
    part = rows.loc[test_days]
    result = {
        # The first three rungs cost nothing: no training at all.
        "1 persistence": scores(rows, baselines.persistence(part), test_days) | {"one fit (ms)": 0.0},
        # Rung 2 is the target baseline: the trend factor frozen on the validation days.
        "2 trend-adjusted persistence": scores(rows, baselines.target_baseline(part), test_days) | {"one fit (ms)": 0.0},
        "3 persistence + average change": scores(rows, average_change(rows, test_days), test_days) | {"one fit (ms)": 0.0},
    }
    for rung, name in (("4 Ridge, water only", "water_only"), ("5 Ridge, all features", "full")):
        columns = train.FEATURE_SETS[name]
        start = time.perf_counter()
        forecast = train.walk_forward(rows, columns, alphas[name], test_days)
        seconds = time.perf_counter() - start
        result[rung] = scores(rows, forecast, test_days) | {
            "one fit (ms)": 1000 * fit_seconds(rows, columns, alphas[name]),
            # One model per test day: the whole walk-forward test.
            "walk-forward (s)": seconds,
        }
    return pd.DataFrame(result).T


def intervals(rows: pd.DataFrame, test_days: pd.DatetimeIndex, alphas: dict) -> pd.DataFrame:
    """Question 1, continued: the differences between rungs, with their 95 % intervals."""
    # Why: the README says e.g. "against row 3 the model gains 6 %" and gives an
    # interval. Before, these intervals were computed by hand; now this function
    # prints them (found in a review on 15.11.2026). The forecasts are the same as in
    # ladder(); they are made again here (a few seconds), so ladder() stays simple.
    part = rows.loc[test_days]
    forecasts = {
        "1 persistence": baselines.persistence(part),
        "2 target baseline": baselines.target_baseline(part),
        "3 average change": average_change(rows, test_days),
        "4 water only": train.walk_forward(rows, train.FEATURE_SETS["water_only"], alphas["water_only"], test_days),
        "5 all features": train.walk_forward(rows, train.FEATURES_FULL, alphas["full"], test_days),
    }
    # Every rung needs a real forecast on every test day, as in scores() (a NaN would
    # quietly drop out of the means and the intervals).
    for forecast in forecasts.values():
        train.check_every_day_forecast(forecast, test_days)
    # The error of each rung on each test day (°C, without sign). reindex: the same
    # days in the same order for every rung, so the yes/no lists of the windows fit.
    errors = {name: (f.reindex(part.index) - part[features.LABEL]).abs() for name, f in forecasts.items()}
    # The comparisons of the README: how far is the model from the target baseline of
    # the success criterion (row 2) and from plain persistence, is row 3 better than
    # persistence (it would pass the gate without any feature), do the features beat
    # row 3, and does the weather help on top of the water features?
    pairs = [
        ("5 all features", "2 target baseline"),
        ("5 all features", "1 persistence"),
        ("3 average change", "1 persistence"),
        ("5 all features", "3 average change"),
        ("5 all features", "4 water only"),
    ]
    result = {}
    for a, b in pairs:
        for window, row in window_differences(errors[a], errors[b], part).items():
            result[(f"{a} - {b}", window)] = row
    return pd.DataFrame(result).T


def learning_curve(rows: pd.DataFrame, test_days: pd.DatetimeIndex, alpha: float) -> pd.DataFrame:
    """Question 2: the full model trained on less data, on the same test days."""
    result = {}
    for name, window in WINDOWS.items():
        forecast = train.walk_forward(rows, train.FEATURES_FULL, alpha, test_days, window_days=window)
        result[name] = scores(rows, forecast, test_days) | {"days with a forecast": len(forecast)}
    return pd.DataFrame(result).T


def retrain_replay(rows: pd.DataFrame, test_days: pd.DatetimeIndex, alpha: float) -> pd.DataFrame:
    """Question 3: the full model retrained less often, on the same test days."""
    result = {}
    for name, every in RETRAIN.items():
        forecast = train.walk_forward(rows, train.FEATURES_FULL, alpha, test_days, retrain_every=every)
        result[name] = scores(rows, forecast, test_days)
    return pd.DataFrame(result).T


def forecast_comparison(rows: pd.DataFrame, single: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Question 4: the same model with the weather features from the 06:00 UTC run instead.

    Returns the error of each variant, and the difference between them with its interval.
    """
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
    result, errors = {}, {}
    for name, part in variants.items():
        test_days = part.index[part.index >= baselines.TEST_START]
        # Each variant chooses its own setting on its validation days, as in train.py.
        alpha = train.choose_alpha(part, train.FEATURES_FULL)[0]
        forecast = train.walk_forward(part, train.FEATURES_FULL, alpha, test_days)
        result[name] = scores(part, forecast, test_days) | {"alpha": alpha, "test days": len(forecast)}
        errors[name] = (forecast - part.loc[forecast.index, features.LABEL]).abs()
    # Is the fresher forecast clearly better? 0.438 against 0.440 degC alone does not
    # say it; the interval of the difference does (found in a review on 15.11.2026).
    # Both variants have the same rows, so they forecast the same days.
    old, fresh = variants  # the two names, in the order of the dict above
    part = rows.loc[errors[old].index]
    differences = window_differences(errors[fresh], errors[old], part)
    return pd.DataFrame(result).T, pd.DataFrame(differences).T


def lake_weather(start: dt.date, end: dt.date) -> pd.DataFrame:
    """The five weather features from the 48-hour-old forecast at the lake point (downloads it)."""
    # The same request as the daily job (ingest_weather_runs.fetch), only at another
    # point, and with the grid cell over the water (location.py); nothing is saved.
    # Public API, no key; about 15 seconds since 2024.
    hourly = ingest_weather_runs.fetch(
        start,
        end,
        location.LAKE_LATITUDE,
        location.LAKE_LONGITUDE,
        cell_selection=location.LAKE_CELL_SELECTION,
    )
    hours = pd.DataFrame(hourly)
    hours["time_utc"] = pd.to_datetime(hours.pop("time"), utc=True)
    hours["time_local"] = hours["time_utc"].dt.tz_convert(features.TIMEZONE)
    # The shared definition of the weather features, on the lake's hours.
    return features.weather_features(hours)


def lake_point_comparison(rows: pd.DataFrame, lake: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Question 5 (MS2 review point B2): is the Geneva point enough, or does a point on the lake help?

    lake: the weather features at the lake point (lake_weather). The validation MAE
    decides; the test days are only reported. Returns the error of each version, and
    the difference of each lake version to Geneva with its interval, on the validation
    year and in each window of the test days.
    """
    lake = lake.reindex(rows.index)
    # Version 2: all five weather features from the lake point instead of Geneva.
    instead = rows.copy()
    instead[features.WEATHER_FEATURES] = lake[features.WEATHER_FEATURES]
    # Version 3: Geneva, plus the lake's south-west wind of today and tomorrow, the two
    # features of the upwelling (review point A3).
    extra = ["wind_sw_lake_d0", "wind_sw_lake_d1"]
    added = rows.copy()
    added[extra] = lake[["wind_sw_fc_d0", "wind_sw_fc_d1"]].to_numpy()
    # A fair comparison: only the days on which both points have all features.
    both = lake[features.WEATHER_FEATURES].notna().all(axis=1)
    variants = {
        "Geneva (the features now)": (rows[both], train.FEATURES_FULL),
        "lake point instead of Geneva": (instead[both], train.FEATURES_FULL),
        "Geneva + the lake's south-west wind": (added[both], train.FEATURES_FULL + extra),
    }
    result, errors, validation_errors = {}, {}, {}
    for name, (part, columns) in variants.items():
        test_days = part.index[part.index >= baselines.TEST_START]
        # Each version chooses its own setting on its validation days, as in train.py,
        # and its validation MAE at that setting is the number that decides.
        alpha, validation = train.choose_alpha(part, columns)
        # The validation forecasts once more at the chosen setting (the same days as in
        # choose_alpha), so the deciding difference gets its interval, and the big-change
        # days of the validation year can be shown too (review of 26.11.2026).
        on_validation = train.walk_forward(part, columns, alpha, baselines.validation_days(part).index)
        validation_errors[name] = (on_validation - part.loc[on_validation.index, features.LABEL]).abs()
        big_on_validation = baselines.big_change_days(part.loc[on_validation.index]).to_numpy()
        forecast = train.walk_forward(part, columns, alpha, test_days)
        result[name] = {
            "validation MAE (decides)": validation[alpha],
            "validation change > 0.8 MAE": validation_errors[name][big_on_validation].mean(),
        } | scores(part, forecast, test_days) | {"alpha": alpha, "test days": len(forecast)}
        errors[name] = (forecast - part.loc[forecast.index, features.LABEL]).abs()
    # Each lake version against Geneva on the same days, with the interval: a small
    # difference alone does not say that one point is better (as in question 4). First
    # the validation year: all its days decide, its big-change days and October to
    # January are shown next to them. Then the test days, which are only reported.
    geneva = next(iter(variants))
    periods = {
        "validation year": (validation_errors, rows.loc[validation_errors[geneva].index]),
        "test days": (errors, rows.loc[errors[geneva].index]),
    }
    differences = {}
    for name in list(variants)[1:]:
        for period, (period_errors, part) in periods.items():
            for window, row in window_differences(period_errors[name], period_errors[geneva], part).items():
                # windows_of() calls the first window "all test days"; here it is all
                # days of the period.
                label = "all days" if window == "all test days" else window
                if period == "validation year" and label == "all days":
                    label += " (decides)"
                differences[f"{name} - Geneva, {period}, {label}"] = row
    return pd.DataFrame(result).T, pd.DataFrame(differences).T


def main() -> None:
    # Command-line options; __doc__ (the text at the top) becomes the --help text.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--features", default="data/features/features.parquet", help="the checked-out feature table (after dvc pull)"
    )
    parser.add_argument(
        "--data-version", metavar="TAG_OR_COMMIT", help="use an older data version, e.g. data-v1 (needs the bucket key)"
    )
    # Question 4 needs the raw files of the 06:00 UTC runs: they are not in the feature
    # store, because the model does not use them (yet).
    parser.add_argument("--raw", default="data/raw", help="folder with the raw files")
    # Question 5 downloads the forecast at the lake point (public API, no key), so it
    # needs the internet and runs only when asked for.
    parser.add_argument("--lake-point", action="store_true", help="also question 5: a weather point on the lake")
    args = parser.parse_args()

    # The same read path as train.py (feature_store.py, review point E1).
    table, data = feature_store.load_features(args.data_version, args.features)
    print(
        f"data version: {data['data_requested']}, tag {data['data_tag']}, DVC hash {data['data_md5']}, "
        f"proven by DVC: {data['data_matches_dvc']}"
    )
    rows = train.model_rows(table)
    test_days = rows.index[rows.index >= baselines.TEST_START]
    # The setting of each model, chosen on the validation year only (as in train.py).
    alphas = {name: train.choose_alpha(rows, columns)[0] for name, columns in train.FEATURE_SETS.items()}
    print(f"{len(test_days)} test days from {test_days.min():%d.%m.%Y}; alpha {alphas}; MAE in degC")

    print("\n1 Ladder: is the model worth its training?")
    print(ladder(rows, test_days, alphas).round(3).to_string())
    print("\n  Differences between rungs (a - b, below 0: a is better) with 95 % intervals")
    print(intervals(rows, test_days, alphas).round(3).to_string())
    print("\n2 Data: the full model trained on less data")
    print(learning_curve(rows, test_days, alphas["full"]).round(3).to_string())
    print("\n3 Retrain: the full model retrained less often")
    print(retrain_replay(rows, test_days, alphas["full"]).round(3).to_string())
    print("\n4 Forecast: the 48-hour-old forecast against the 06:00 UTC run of the same day")
    single = features.single_run_weather_features(features.load_single_runs(args.raw))
    comparison, differences = forecast_comparison(rows, single)
    print(comparison.round(3).to_string())
    print("\n  06:00 UTC run - 48-hour-old forecast (below 0: the fresher one is better) with 95 % intervals")
    print(differences.round(3).to_string())
    if args.lake_point:
        print(f"\n5 Weather point: Geneva against a point on the lake ({location.LAKE_LATITUDE} N, {location.LAKE_LONGITUDE} E)")
        # One day before the first row (for the night hours of the first day) up to tomorrow.
        start, end = rows.index.min() - pd.Timedelta(days=1), table.index.max() + pd.Timedelta(days=1)
        lake = lake_weather(start.date(), end.date())
        comparison, differences = lake_point_comparison(rows, lake)
        print(comparison.round(3).to_string())
        print(
            "\n  Lake version - Geneva (below 0: the lake version is better) with 95 % intervals:"
            " the validation year decides, the test days are only reported"
        )
        print(differences.round(3).to_string())


# Runs main() only when the file is started as a script, not when it is imported
# (the tests import this file).
if __name__ == "__main__":
    main()
