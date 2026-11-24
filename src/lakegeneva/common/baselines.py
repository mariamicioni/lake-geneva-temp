"""The two baselines the model has to beat, and how they are compared.

- persistence: tomorrow's mean = the last reading before 18:00 today
- trend-adjusted persistence: the same plus a factor times the 3-day trend
  (suggested by both MS1 reviewers, points A2/B2). The factor was chosen once, on
  the validation days, and is frozen: this is the baseline of the success criterion.

    python src/lakegeneva/common/baselines.py    # prints the comparison table
"""
# Why in common/: the training pipeline compares every model with these baselines,
# and the live system will show the model's error next to the error of
# persistence. Both use this file, so the baselines are defined only once.
# Why "the last reading" and not "today's mean until 18:00": it is the newest
# information at 18:00 ("tomorrow = the temperature now"), and it is also the
# better of the two over the whole year (main() prints both).

import numpy as np  # random numbers and percentiles for the bootstrap
import pandas as pd  # the daily table

from lakegeneva.common import features  # the shared feature table and its column names

# --- Settings -----------------------------------------------------------------
# Days on which tomorrow's mean differs by more than 0.8 °C from the last
# reading: the days where persistence fails (proposal, review point B3).
BIG_CHANGE = 0.8
# Success criterion: at least 20 % lower MAE than the target baseline (below), on
# the same days.
TARGET_GAIN = 0.20
# The test days (review point A1): the first year with all features (21.01.2024
# to 20.01.2025) is for validation, to choose the model settings in MS3, so it
# never counts for the target. Only the days from here on count, and on each of
# them the model is trained again on the earlier days whose label is already
# known at 18:00 (walk-forward). The day just before is not used: its label is
# the mean of the test day itself, and that day is not over at 18:00.
TEST_START = pd.Timestamp("2025-01-21")
# The first day with all features: the validation days run from here to 19.01.2025.
VALIDATION_START = pd.Timestamp("2024-01-21")

# --- The target baseline: chosen on the validation days, then frozen ------------
# Before, the target was 20 % below the STRONGER of the two baselines on each
# window's own test days. Then the choice could switch as new test days came in,
# and the target moved with it (MS2 editorial summary, point B1). Now the baseline
# and its trend factor were chosen ONCE, on the validation days only, on 20.11.2026:
# - the days: 21.01.2024 to 19.01.2025, every day with all features and a label
#   (365 days). 20.01.2025 is left out: its label is the mean of 21.01.2025, the
#   first test day (the same rule as train.known_at_18).
# - the factor: the one with the lowest MAE on a grid from -1 to +1 in steps of
#   0.005 (fit_trend_factor() below; the MAE because the target is an MAE).
# Result: factor -0.075, MAE 0.4949 degC, against 0.5062 degC for plain persistence
# (factor 0) and 0.7459 degC for the factor 1/3 of the MS1 reviews. So this is the
# stronger baseline on the validation days, and it is the target baseline from now on.
# The factor is NEGATIVE: tomorrow tends to move a little AGAINST the 3-day trend
# ("mean-reverting persistence"). The gain over persistence is small (95 % interval
# -0.0245 to -0.0001 degC) and the factor was fitted on those same days, so this is no
# proof that it is better on new days. It is a fixed choice: the target never
# switches back to plain persistence, also when persistence is better on some test
# days. Plain persistence stays in every table, for comparison.
# `python src/lakegeneva/common/baselines.py` fits the factor again, as a check.
TREND_FACTOR = -0.075
# The factors that fit_trend_factor() tries: -1.000, -0.995, ..., +1.000 (0 included).
TREND_GRID = np.round(np.arange(-1, 1.0001, 0.005), 3)

# Bootstrap: blocks of 7 days, 2,000 new test sets, a fixed seed (the same
# interval on every run).
BLOCK_DAYS = 7
N_RESAMPLES = 2000
SEED = 0
# An interval needs at least 5 blocks (35 days). With fewer days the new test sets
# are almost all the same, and the interval looks far too sure (checked on
# 27.10.2026: 8 days gave a narrow interval that means nothing).
MIN_DAYS_BOOTSTRAP = 5 * BLOCK_DAYS


def persistence(table: pd.DataFrame) -> pd.Series:
    """Tomorrow's mean = the last reading before 18:00 today."""
    return table["water_last"]


def trend_adjusted(table: pd.DataFrame, factor: float = TREND_FACTOR) -> pd.Series:
    """Persistence plus `factor` times the 3-day trend (by default the frozen factor)."""
    # water_trend_3d is the change over three days. The MS1 reviews suggested a third
    # of it (one day of the trend); the factor fitted on the validation days is -0.075.
    return table["water_last"] + factor * table["water_trend_3d"]


def target_baseline(table: pd.DataFrame) -> pd.Series:
    """The baseline of the success criterion: trend-adjusted persistence with the frozen factor."""
    # One function, so train.py, experiments.py and this file all use the same one.
    return trend_adjusted(table, TREND_FACTOR)


def target_mae(baseline_mae: float) -> float:
    """What the model must reach: 20 % below the MAE of the target baseline on the same days."""
    return (1 - TARGET_GAIN) * baseline_mae


def validation_days(table: pd.DataFrame) -> pd.DataFrame:
    """The rows on which the target baseline was chosen: all features and a label, before the test days."""
    complete = table[features.FEATURES + [features.LABEL]].notna().all(axis=1)
    # Up to two days before TEST_START: the label of the day before is the mean of
    # the first test day, which is not over at 18:00 on that day.
    last_day = TEST_START - pd.Timedelta(days=2)
    return table[complete & (table.index >= VALIDATION_START) & (table.index <= last_day)]


def fit_trend_factor(rows: pd.DataFrame) -> tuple[float, pd.Series]:
    """The trend factor with the lowest MAE on `rows`, and the MAE of every factor tried."""
    # How TREND_FACTOR was found. Only for the validation days: fitted on test days,
    # the baseline would again depend on the days it is judged on.
    errors = pd.Series(
        {factor: (trend_adjusted(rows, factor) - rows[features.LABEL]).abs().mean() for factor in TREND_GRID}
    )
    # idxmin: the factor of the smallest error (the first one if two are equal).
    return float(errors.idxmin()), errors


def big_change_days(table: pd.DataFrame) -> pd.Series:
    """True for the days on which tomorrow's mean differs by more than 0.8 °C from the last reading."""
    return (table[features.LABEL] - table["water_last"]).abs() > BIG_CHANGE


def bootstrap_mae_difference(errors_a, errors_b, block: int = BLOCK_DAYS, seed: int = SEED):
    """MAE(a) - MAE(b) and its 95 % confidence interval (moving-block bootstrap of calendar days)."""
    # The bootstrap answers "how sure are we about this difference?": it builds many
    # new test sets from the days we have (drawing with replacement) and looks at
    # how much the difference moves between them. Days are drawn in blocks of 7
    # calendar days, because one windy week affects several days, so single days
    # are not independent (moving-block bootstrap, Künsch 1989, [8] in README.md).
    # Why calendar days and not 7 rows: for a part of the days (e.g. only the
    # big-change days) 7 rows in a row can be months apart (up to 228 days on
    # 30.10.2026), so they are no "week". A block is now 7 real days, and it holds
    # only the days of the part that fall into that week (0 to 7 days). Promised in
    # the MS2 summary; the effect on the intervals is in the commit message.
    # The errors come with their dates (a pandas Series). A plain list of numbers
    # counts as consecutive days.
    errors_a, errors_b = pd.Series(errors_a, dtype=float), pd.Series(errors_b, dtype=float)
    if isinstance(errors_a.index, pd.DatetimeIndex):
        dates = errors_a.index
    else:
        dates = pd.date_range("2000-01-01", periods=len(errors_a), freq="D")
    # Per day: how much larger the error of a was than the error of b.
    diff = errors_a.to_numpy() - errors_b.to_numpy()

    # Each day as a number: 0 for the first day, 1 for the next calendar day, ...
    offset = (dates - dates.min()).days.to_numpy()
    span = int(offset.max()) + 1  # calendar days from the first to the last day
    block = min(block, span)  # a part shorter than one block is one block
    # Sum and count of the differences on every calendar day (0 on days not in the part).
    day_sum = np.bincount(offset, weights=diff, minlength=span)
    day_count = np.bincount(offset, minlength=span)
    # Sum and count of every possible block of 7 calendar days, from its first day s:
    # with cumulative sums, block s is "everything up to s + 7" minus "everything up to s".
    cum_sum = np.concatenate([[0.0], np.cumsum(day_sum)])
    cum_count = np.concatenate([[0], np.cumsum(day_count)])
    block_sum = cum_sum[block:] - cum_sum[:-block]
    block_count = cum_count[block:] - cum_count[:-block]

    rng = np.random.default_rng(seed)  # fixed seed: the same result every run
    n_blocks = int(np.ceil(span / block))  # blocks needed to cover the calendar span
    # A random first day for every block of every new test set.
    starts = rng.integers(0, span - block + 1, size=(N_RESAMPLES, n_blocks))
    # The mean difference of each new test set: all its days together, so a block
    # with more days of the part counts more.
    totals = block_count[starts].sum(axis=1)
    means = block_sum[starts].sum(axis=1) / np.where(totals > 0, totals, np.nan)
    # 95 % of the new test sets give a difference between these two values. A new
    # test set without any day (only possible for a tiny part) is left out.
    low, high = np.nanpercentile(means, [2.5, 97.5])
    return diff.mean(), low, high


def report(table: pd.DataFrame) -> pd.DataFrame:
    """Both baselines on the same days, for every window the milestone summary needs."""
    # Only days on which the model can be tested too: all features and a label.
    all_features = table[features.FEATURES + [features.LABEL]].notna().all(axis=1)
    # The water features reach back to 2016 (the weather features to 21.01.2024).
    water_only = table[features.WATER_FEATURES + [features.LABEL]].notna().all(axis=1)
    # October to January, by the day that is forecast (tomorrow), as in the proposal.
    forecast_month = (table.index + pd.Timedelta(days=1)).month
    oct_jan = pd.Series(forecast_month.isin([10, 11, 12, 1]), index=table.index)
    big = big_change_days(table)
    # The days on which the target baseline was chosen (see TREND_FACTOR).
    validation = pd.Series(table.index.isin(validation_days(table).index), index=table.index)
    # The days that count for the success criterion: all features, from TEST_START on.
    test = all_features & (table.index >= TEST_START)
    windows = {
        "validation, all year": validation,
        "validation, change > 0.8": validation & big,
        "test, all year": test,
        "test, Oct-Jan": test & oct_jan,
        "test, change > 0.8": test & big,
        "test, Oct-Jan, change > 0.8": test & oct_jan & big,
        # The same windows over all days with all features (validation year included).
        "all features, all year": all_features,
        "all features, Oct-Jan": all_features & oct_jan,
        "all features, change > 0.8": all_features & big,
        "all features, Oct-Jan, change > 0.8": all_features & oct_jan & big,
        "water only, all year": water_only,
        "water only, Oct-Jan": water_only & oct_jan,
    }

    rows = []
    for name, in_window in windows.items():
        part = table[in_window]
        # The error of each baseline on each day (°C, without sign).
        error_persistence = (persistence(part) - part[features.LABEL]).abs()
        error_trend = (target_baseline(part) - part[features.LABEL]).abs()
        row = {"window": name, "days": len(part)}
        if len(part) > 0:
            mae_persistence, mae_trend = error_persistence.mean(), error_trend.mean()
            # |= adds these entries to the row dict.
            row |= {
                "first_day": part.index.min().date(),
                "persistence": mae_persistence,
                "trend_adjusted": mae_trend,
                "trend_minus_persistence": mae_trend - mae_persistence,
                # What the model must reach: 20 % below the TARGET baseline, always the
                # same one (no longer the better of the two on these days).
                "target_mae": target_mae(mae_trend),
            }
        if len(part) >= MIN_DAYS_BOOTSTRAP:
            # The first value (the difference) is already in the row, so "_" skips it.
            _, row["ci_low"], row["ci_high"] = bootstrap_mae_difference(error_trend, error_persistence)
        rows.append(row)
    return pd.DataFrame(rows).set_index("window")


def main() -> None:
    table = features.feature_table("data/raw")
    result = report(table)
    # Plain "degC" instead of the degree sign, which some Windows terminals cannot show.
    print("MAE in degC at the 18:00 cutoff; trend - persistence with its 95 % bootstrap interval (ci_low, ci_high)")
    print(result.round(3).to_string())
    # For comparison: persistence with today's mean until 18:00 instead of the last
    # reading (the reason for choosing the last reading, see the top of this file).
    days = table[features.FEATURES + [features.LABEL]].notna().all(axis=1)
    part = table[days]
    mean_d0 = (part["water_mean_d0"] - part[features.LABEL]).abs().mean()
    last = (persistence(part) - part[features.LABEL]).abs().mean()
    print(f"\nall features, all year: persistence with the last reading {last:.3f}, with today's mean until 18:00 {mean_d0:.3f}")
    # The choice of the target baseline, fitted again on the validation days: if the
    # result is not TREND_FACTOR any more, the old data changed and a person must look.
    validation = validation_days(table)
    factor, errors = fit_trend_factor(validation)
    plain = (persistence(validation) - validation[features.LABEL]).abs().mean()
    ms1 = (trend_adjusted(validation, 1 / 3) - validation[features.LABEL]).abs().mean()
    print(
        f"\ntarget baseline, fitted on {len(validation)} validation days "
        f"({validation.index.min().date()} to {validation.index.max().date()}): "
        f"persistence {plain:.4f}, factor 1/3 {ms1:.4f}, best factor {factor:+.3f} -> {errors[factor]:.4f} degC; "
        f"frozen factor {TREND_FACTOR:+.3f}" + ("" if factor == TREND_FACTOR else " <- DIFFERENT, check the data")
    )


# Runs main() only when the file is started as a script, not when it is imported.
if __name__ == "__main__":
    main()
