"""Training pipeline: read the feature store, train the model and test it day by day.

The model is a Ridge regression. It predicts tomorrow's CHANGE (tomorrow's mean
minus the last reading before 18:00), and the forecast is the last reading plus
that change. It is tested on the test days from 21.01.2025 the way it will be used
live: for every test day it is trained again on the earlier days only (walk-forward).

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

from lakegeneva.common import baselines, features  # the shared definitions

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
            if n - start < MIN_TRAIN_ROWS:
                continue  # too few earlier rows to learn anything: no forecast for this day
            model = make_model(alpha).fit(x[start:n], y[start:n])
            trained_on = day
        # A kept model was trained on day trained_on with the rows up to trained_on - 2,
        # which are also known on every later day, so keeping it never leaks.
        i = rows.index.get_loc(day)  # the position of the day to forecast
        # x[i : i + 1] keeps a table with one row, which predict() needs.
        forecasts[day] = last[i] + model.predict(x[i : i + 1])[0]
    return pd.Series(forecasts, name="forecast", dtype=float).rename_axis("date")


def mae(rows: pd.DataFrame, forecast: pd.Series) -> float:
    """Mean absolute error of a forecast on its days (°C)."""
    return float((forecast - rows.loc[forecast.index, features.LABEL]).abs().mean())


def choose_alpha(rows: pd.DataFrame, columns: list[str]) -> tuple[float, dict]:
    """The penalty with the lowest walk-forward error on the validation year."""
    # Only days before TEST_START: the test days never decide a setting of the model.
    validation_days = rows.index[rows.index < baselines.TEST_START]
    errors = {alpha: mae(rows, walk_forward(rows, columns, alpha, validation_days)) for alpha in ALPHAS}
    # min(..., key=errors.get): the alpha whose error is the smallest.
    return min(errors, key=errors.get), errors


def evaluate(rows: pd.DataFrame, forecast: pd.Series) -> pd.DataFrame:
    """The model against persistence on the test days, in the windows of the success criterion."""
    part = rows.loc[forecast.index]
    error_model = (forecast - part[features.LABEL]).abs()
    error_persistence = (baselines.persistence(part) - part[features.LABEL]).abs()
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
            row |= {
                "model": mae_model,
                "persistence": mae_persistence,
                # The success criterion: 20 % below persistence on the same days.
                "target": (1 - baselines.TARGET_GAIN) * mae_persistence,
                "gain_%": 100 * (1 - mae_model / mae_persistence),
            }
        if row["days"] >= baselines.MIN_DAYS_BOOTSTRAP:
            # model - persistence: below 0 means the model is better. The interval
            # says how sure that is (moving-block bootstrap, as for the baselines).
            row["diff"], row["ci_low"], row["ci_high"] = baselines.bootstrap_mae_difference(
                error_model[in_window], error_persistence[in_window]
            )
        result.append(row)
    return pd.DataFrame(result).set_index("window")


def train_final(rows: pd.DataFrame, columns: list[str], alpha: float):
    """The model for the live forecast: trained on every row with a known label."""
    # The feature table only has a label once tomorrow is over, so every row here is
    # already known. On the evening of day D the newest row with a label is D-2, so
    # this is exactly the model that the walk-forward test uses for day D.
    # Plain arrays, like in walk_forward(), so the final model is the same kind of model.
    return make_model(alpha).fit(rows[columns].to_numpy(), change(rows).to_numpy())


def run(table: pd.DataFrame) -> dict:
    """Choose the setting, test walk-forward and train the final model, for both feature sets."""
    rows = model_rows(table)
    test_days = rows.index[rows.index >= baselines.TEST_START]
    results = {}
    for name, columns in FEATURE_SETS.items():
        alpha, validation_errors = choose_alpha(rows, columns)
        forecast = walk_forward(rows, columns, alpha, test_days)
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


# Runs main() only when the file is started as a script, not when it is imported
# (the tests import this file).
if __name__ == "__main__":
    main()
