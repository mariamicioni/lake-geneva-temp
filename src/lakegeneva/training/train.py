"""Training pipeline: read the feature store, train the model and test it day by day.

The model is a Ridge regression. It predicts tomorrow's CHANGE (tomorrow's mean
minus the last reading before 18:00), and the forecast is the last reading plus
that change (common/forecasting.py, shared with the live forecast). It is tested on
the test days from 21.01.2025 the way it will be used live: for every test day it is
trained again on the earlier days only (walk-forward).

    python src/lakegeneva/training/train.py
"""
# Why Ridge: a linear model with ten features and about 1,000 rows. It trains in a
# millisecond, so it can be trained again for every single test day, and every
# coefficient can be read and explained. "Ridge" adds a penalty on large
# coefficients; its strength (alpha) is the one setting chosen on the validation year.
# Why the change and not the temperature itself: the temperature tomorrow is almost
# the temperature now (persistence is a strong baseline). Predicting only the change
# lets the model start from persistence: with a very strong penalty all coefficients
# shrink to 0 and the forecast is the last reading plus the average change.
# Why the features are scaled (StandardScaler): the penalty treats all coefficients
# the same, so all features must have the same scale (°C, m³/s and hours otherwise).
# The scaler learns its mean and spread from the training rows only.

import argparse  # reads the command-line option --features

import numpy as np  # np.ones: the yes/no list of the "all year" window
import pandas as pd  # the daily feature table
from sklearn.linear_model import Ridge  # linear regression with a penalty
from sklearn.pipeline import make_pipeline  # scaling and regression as one model
from sklearn.preprocessing import StandardScaler  # every feature to mean 0, spread 1

# The shared definitions: the baselines, the features, and the step from the model's
# answer (a change) to the forecast (a temperature).
from lakegeneva.common import baselines, features, forecasting

# --- Settings -----------------------------------------------------------------
# The two models: all ten features, and only the five water features (review point
# B2: "a pass would not show that the weather features add anything"). Both are
# trained and tested on the same rows, so only the features differ.
FEATURE_SETS = {"full": features.FEATURES, "water_only": features.WATER_FEATURES}
# The features of the full model (the one for the live forecast).
FEATURES_FULL = FEATURE_SETS["full"]
# The penalty strengths that are tried on the validation year (from weak to strong).
# The range is wide on purpose: on 14.11.2026 the best value of a first, shorter
# list (0.1 to 1,000) was its largest one, so the best value could lie beyond it.
ALPHAS = (0.1, 1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)
# A validation day counts only if at least 90 earlier rows (about three months) can
# be used for training. The first days of the validation year have almost none.
MIN_TRAIN_ROWS = 90


def model_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The rows a model can learn from and be tested on: all ten features and a label."""
    # The same rows for both models (also the water-only one), so the comparison is fair.
    complete = table[features.FEATURES + [features.LABEL]].notna().all(axis=1)
    return table[complete].sort_index()


def known_at_18(rows: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """The rows whose label is already known at 18:00 on `day`: up to the day before yesterday."""
    # The label of a row is the mean of the NEXT day. At 18:00 on day D:
    # - the label of D-2 is the mean of D-1, which is over: known;
    # - the label of D-1 is the mean of D itself, which is not over: NOT known.
    # So a model for day D may only learn from rows up to D-2. This is the main rule
    # against leakage in training (tests/test_train.py checks it).
    return rows[rows.index <= day - pd.Timedelta(days=2)]


def make_model(alpha: float):
    """A fresh, untrained model: scale the features, then Ridge regression."""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def change(rows: pd.DataFrame) -> pd.Series:
    """What the model learns: tomorrow's mean minus the last reading before 18:00."""
    # The way back, from a predicted change to a temperature, is
    # forecasting.temperature(): the last reading plus the change.
    return rows[features.LABEL] - rows["water_last"]


def walk_forward(
    rows: pd.DataFrame,
    columns: list[str],
    alpha: float,
    days,
    window_days: int | None = None,
    retrain_every: int = 1,
) -> pd.Series:
    """Forecast every day in `days`, each with a model trained only on what was known at 18:00.

    window_days: learn only from the rows of the last window_days days (None: all rows).
    retrain_every: train a new model only every this many days and keep using the last
    one in between (1: a new model every day).
    A day gets no forecast only if no model exists yet. If a new model is due but its
    window has too few rows (a gap in the data), the last model is kept.
    """
    # The two options are for experiments.py (how much data, how often to retrain).
    # train.py uses the defaults: all earlier rows, a new model for every day.
    # Plain arrays instead of the table: the loop below trains hundreds of models, and
    # cutting an array is much faster than filtering a table (the same rows either way).
    x = rows[columns].to_numpy()
    y = change(rows).to_numpy()
    last = rows["water_last"].to_numpy()
    forecasts = {}
    model, trained_on = None, None
    for day in days:
        # A new model on the first day, and again once retrain_every days have passed.
        if model is None or (day - trained_on).days >= retrain_every:
            # The rows are sorted by date, so the rows known at 18:00 are the first n rows.
            n = len(known_at_18(rows, day))
            # With a window, the training rows start window_days before the day:
            # searchsorted finds the position of the first row after that date.
            start = 0
            if window_days is not None:
                start = rows.index.searchsorted(day - pd.Timedelta(days=window_days), side="right")
            if n - start >= MIN_TRAIN_ROWS:
                model = make_model(alpha).fit(x[start:n], y[start:n])
                trained_on = day
            elif model is None:
                continue  # too few earlier rows and no older model: no forecast for this day
            # Otherwise the window has too few rows (a gap in the data), but an older model
            # exists: keep using it, as the live system keeps the champion until a new
            # model is ready. Before, the day was skipped, so lines of experiments.py
            # could be on different days without any error (found in a review on
            # 15.11.2026; no effect on today's numbers, every line had 661 days).
            # trained_on stays the same, so the next day tries a new model again.
        # A kept model was trained on day trained_on with the rows up to trained_on - 2,
        # which are also known on every later day, so keeping it never leaks.
        i = rows.index.get_loc(day)  # the position of the day to forecast
        # The forecast is the last reading plus the predicted change. This step is not
        # written here: it is forecasting.temperature(), the same function that the
        # live forecast (MS4) will use on the registered model. So the forecast that
        # is tested here and the live forecast cannot differ in this step
        # (training-serving skew). Before, the step was a line of its own here:
        # last[i] + model.predict(x[i : i + 1])[0], and the numbers are the same
        # (checked on 15.11.2026: every forecast on the validation days is equal, bit by bit).
        # x[i : i + 1] and last[i : i + 1] keep one row, which predict() needs; the
        # result is a list with one forecast, and [0] takes it out.
        forecasts[day] = forecasting.temperature(model, x[i : i + 1], last[i : i + 1])[0]
    return pd.Series(forecasts, name="forecast", dtype=float).rename_axis("date")


def mae(rows: pd.DataFrame, forecast: pd.Series) -> float:
    """Mean absolute error of a forecast on its days (°C)."""
    return float((forecast - rows.loc[forecast.index, features.LABEL]).abs().mean())


def choose_alpha(rows: pd.DataFrame, columns: list[str]) -> tuple[float, dict]:
    """The penalty with the lowest walk-forward error on the validation year."""
    # The setting must be fixed before the first test forecast (18:00 on TEST_START),
    # so it may only use labels that are known then: the rows up to TEST_START - 2.
    # The row of TEST_START - 1 (20.01.2025) is left out: its label is the mean of the
    # first test day itself, which is not over at 18:00 (found in a review on
    # 14.11.2026; alpha stayed 1000 for both models). The test days never count either.
    validation_days = known_at_18(rows, baselines.TEST_START).index
    errors = {alpha: mae(rows, walk_forward(rows, columns, alpha, validation_days)) for alpha in ALPHAS}
    # If no validation day has MIN_TRAIN_ROWS earlier rows, no day gets a forecast and
    # every error is empty (NaN). min() below would then quietly return the first alpha
    # of the list (0.1, the weakest penalty: 0.499 instead of 0.437 degC on the test
    # days), and the run would be logged as if all was normal. So stop with a clear
    # message instead (found in a review on 15.11.2026). This happens with a table
    # that starts too late, e.g. the no-key path of the README with a later --start.
    if all(pd.isna(error) for error in errors.values()):
        raise ValueError(
            f"alpha cannot be chosen: no validation day before {baselines.TEST_START:%d.%m.%Y} has "
            f"{MIN_TRAIN_ROWS} earlier rows with all features and a label. Start the table earlier."
        )
    # min(..., key=errors.get): the alpha whose error is the smallest.
    return min(errors, key=errors.get), errors


def check_every_day_forecast(forecast: pd.Series, days) -> None:
    """Stop with an error unless every day in `days` has a forecast that is a real number."""
    # Why: the MAE is a mean, and pandas' mean() quietly skips a missing value (NaN).
    # A day without a forecast would then simply not count. In a review on 18.11.2026,
    # setting the 8 worst forecasts to NaN made the 20 % target flip from "missed" to
    # "met", while the table still said 661 days. So the result is only computed if
    # every expected day has a forecast.
    days = pd.DatetimeIndex(days)
    missing = days.difference(forecast.index)
    not_a_number = forecast.index[~np.isfinite(forecast.to_numpy(dtype=float))]
    if len(missing) or len(not_a_number):
        raise ValueError(
            f"{len(missing)} of {len(days)} days have no forecast and {len(not_a_number)} forecasts "
            "are not a number: no MAE and no target are computed."
        )


def evaluate(rows: pd.DataFrame, forecast: pd.Series) -> pd.DataFrame:
    """The model against the two baselines on the test days, in the windows of the success criterion."""
    # Every day of the forecast must be a real number (no NaN, no infinity).
    check_every_day_forecast(forecast, forecast.index)
    part = rows.loc[forecast.index]
    error_model = (forecast - part[features.LABEL]).abs()
    error_persistence = (baselines.persistence(part) - part[features.LABEL]).abs()
    # The target baseline: trend-adjusted persistence with the factor that was chosen
    # on the validation days and frozen (baselines.TREND_FACTOR, MS2 review point B1).
    error_trend = (baselines.target_baseline(part) - part[features.LABEL]).abs()
    # October to January by the day that is forecast (tomorrow), as in baselines.py.
    forecast_month = (part.index + pd.Timedelta(days=1)).month
    windows = {
        "test, all year": np.ones(len(part), dtype=bool),
        "test, change > 0.8": baselines.big_change_days(part).to_numpy(),
        "test, Oct-Jan": forecast_month.isin([10, 11, 12, 1]),
    }
    result = []
    for name, in_window in windows.items():
        row = {"window": name, "days": int(in_window.sum())}
        if row["days"] > 0:
            mae_model = error_model[in_window].mean()
            mae_persistence = error_persistence[in_window].mean()
            mae_trend = error_trend[in_window].mean()
            row |= {
                "model": mae_model,
                # Plain persistence: only for comparison, it decides nothing.
                "persistence": mae_persistence,
                "trend_adjusted": mae_trend,
                # The success criterion of the README: 20 % below the TARGET baseline
                # on the same days. Always the same baseline: before, it was the
                # better of the two on these days, so it could switch as test days
                # came in (MS2 review point B1). The number itself still moves a
                # little with every new test day; the baseline and its factor do not.
                "target": baselines.target_mae(mae_trend),
                # The gain against the same baseline, so "gain_% >= 20" means the
                # target is met.
                "gain_%": 100 * (1 - mae_model / mae_trend),
            }
        if row["days"] >= baselines.MIN_DAYS_BOOTSTRAP:
            # model - target baseline: below 0 means the model is better. The interval
            # says how sure that is (moving-block bootstrap, as for the baselines).
            row["diff"], row["ci_low"], row["ci_high"] = baselines.bootstrap_mae_difference(
                error_model[in_window], error_trend[in_window]
            )
        result.append(row)
    return pd.DataFrame(result).set_index("window")


def train_final(rows: pd.DataFrame, columns: list[str], alpha: float):
    """The model for the live forecast: trained on every row with a known label."""
    # The feature table only has a label once tomorrow is over, so every row here is
    # already known. Trained in the evening of day D, after the feature pipeline, the
    # newest row with a label would be D-2: the model of the walk-forward test for D.
    # But the weekly job (retrain.yml) runs on Monday at 06:00 UTC with the table of
    # Sunday evening, when Sunday is not over yet. So its newest row with a label is
    # FRIDAY's: the model learns up to D-3 for Monday's forecast and up to D-9 for
    # Sunday's (found in a review on 15.11.2026; before, this comment said "exactly"
    # D-2). This costs nothing: replayed with this real schedule, the test error is
    # 0.436 / 1.544 degC against 0.437 / 1.547 degC with a new model every day.
    # Trained on the table (not plain arrays like in walk_forward()): the model then
    # keeps the column names, and forecasting.temperature_for_rows() uses them to hand
    # the model the right columns in the right order in the live forecast. The numbers
    # are the same either way (tests/test_train.py compares the two).
    # Like every model here, it returns the CHANGE. The live forecast must not use
    # model.predict() as a temperature: it calls forecasting.temperature_for_rows().
    return make_model(alpha).fit(rows[columns], change(rows))


def run(table: pd.DataFrame) -> dict:
    """Choose the setting, test walk-forward and train the final model, for both feature sets."""
    rows = model_rows(table)
    test_days = rows.index[rows.index >= baselines.TEST_START]
    results = {}
    for name, columns in FEATURE_SETS.items():
        alpha, validation_errors = choose_alpha(rows, columns)
        forecast = walk_forward(rows, columns, alpha, test_days)
        # Every test day must have a forecast: walk_forward skips a day when it has too
        # few earlier rows, and such a day must not quietly drop out of the result.
        check_every_day_forecast(forecast, test_days)
        results[name] = {
            "alpha": alpha,
            "validation_mae": validation_errors,
            "forecast": forecast,
            "evaluation": evaluate(rows, forecast),
            "model": train_final(rows, columns, alpha),
            "columns": columns,
        }
    return results


def main() -> None:
    # Command-line options; __doc__ (the text at the top) becomes the --help text.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--features", default="data/features/features.parquet", help="the feature table (the feature store)"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="train and test only, log nothing")
    args = parser.parse_args()

    # The training pipeline reads the stored table, not the raw files: the feature
    # store is the only way data reaches the model.
    table = pd.read_parquet(args.features)
    results = run(table)
    for name, result in results.items():
        print(f"\n=== {name}: alpha {result['alpha']} (validation MAE per alpha, degC: "
              + ", ".join(f"{a}: {e:.3f}" for a, e in result["validation_mae"].items()) + ")")
        print(result["evaluation"].round(3).to_string())
        # The coefficients of the final model, per feature (on the scaled features,
        # so they can be compared: degC of change per one spread of the feature).
        ridge = result["model"][-1]
        coefficients = pd.Series(ridge.coef_, index=result["columns"]).round(3)
        print("coefficients:", coefficients.to_dict())

    if not args.no_mlflow:
        # Log both models with the code and data versions, register the full model and
        # promote it if it passes the gate (track.py, gate.py).
        from lakegeneva.training import gate, track

        print()
        # model_rows: the rows with a label, for training and the validation checks.
        # newest_rows: the newest days with all features, also without a label yet:
        # the days of the live forecast, for the shadow and plausibility checks.
        decision = track.track(results, model_rows(table), args.features, gate.newest_rows(table))
        # A version that did not pass the gate turns the run red (exit code 1), so GitHub
        # sends an email and a person looks at it. The version stays registered with its
        # tags, and the champion stays the same.
        if not decision.get("promoted"):
            raise SystemExit(f"The gate did not promote version {decision.get('version')}: see its gate tags in MLflow.")


# Runs main() only when the file is started as a script, not when it is imported
# (the tests import this file).
if __name__ == "__main__":
    main()
