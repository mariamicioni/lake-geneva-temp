"""The two baselines the model has to beat, and how they are compared.

- persistence: tomorrow's mean = the last reading before 18:00 today
- trend-adjusted persistence: the same plus one day of the recent 3-day trend
  (suggested by both MS1 reviewers, points A2/B2)

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
# Success criterion: at least 20 % lower MAE than the stronger baseline, on the
# same days.
TARGET_GAIN = 0.20
# The test days (review point A1): the first year with all features (21.01.2024
# to 20.01.2025) is for validation, to choose the model settings in MS3, so it
# never counts for the target. Only the days from here on count, and on each of
# them the model is trained again on all days before it (walk-forward).
TEST_START = pd.Timestamp("2025-01-21")
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


def trend_adjusted(table: pd.DataFrame) -> pd.Series:
    """Persistence plus a third of the 3-day trend (the change of one day)."""
    # water_trend_3d is the change over three days, so /3 is the change per day.
    return table["water_last"] + table["water_trend_3d"] / 3


def big_change_days(table: pd.DataFrame) -> pd.Series:
    """True for the days on which tomorrow's mean differs by more than 0.8 °C from the last reading."""
    return (table[features.LABEL] - table["water_last"]).abs() > BIG_CHANGE


def bootstrap_mae_difference(errors_a, errors_b, block: int = BLOCK_DAYS, seed: int = SEED):
    """MAE(a) - MAE(b) and its 95 % confidence interval (moving-block bootstrap)."""
    # The bootstrap answers "how sure are we about this difference?": it builds many
    # new test sets from the days we have (drawing with replacement) and looks at
    # how much the difference moves between them. Days are drawn in blocks of 7
    # days in a row, because one windy week affects several days, so single days
    # are not independent (moving-block bootstrap, Künsch 1989, [8] in README.md).
    # Careful: "in a row" means 7 rows of the table it gets. For a part of the days
    # (e.g. only the big-change days) 7 rows can be far apart: up to 228 calendar days
    # on 30.10.2026. Blocks of real calendar days moved the intervals by at most
    # 0.016 °C and changed no conclusion, so this stays for MS2 and is fixed in MS3.
    # Per day: how much larger the error of a was than the error of b.
    diff = np.asarray(errors_a, dtype=float) - np.asarray(errors_b, dtype=float)
    n = len(diff)
    block = min(block, n)  # a test set shorter than one block is one block
    rng = np.random.default_rng(seed)  # fixed seed: the same result every run
    n_blocks = int(np.ceil(n / block))  # blocks needed to cover n days
    # A random first day for every block of every new test set ...
    starts = rng.integers(0, n - block + 1, size=(N_RESAMPLES, n_blocks))
    # ... the 7 days of each block (start, start + 1, ..., start + 6), all blocks
    # of a test set in one row, cut back to n days.
    days = (starts[:, :, None] + np.arange(block)).reshape(N_RESAMPLES, -1)[:, :n]
    # The mean difference in each of the 2,000 new test sets.
    means = diff[days].mean(axis=1)
    # 95 % of the new test sets give a difference between these two values.
    low, high = np.percentile(means, [2.5, 97.5])
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
    # The days that count for the success criterion: all features, from TEST_START on.
    test = all_features & (table.index >= TEST_START)
    windows = {
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
        error_trend = (trend_adjusted(part) - part[features.LABEL]).abs()
        row = {"window": name, "days": len(part)}
        if len(part) > 0:
            mae_persistence, mae_trend = error_persistence.mean(), error_trend.mean()
            # |= adds these entries to the row dict.
            row |= {
                "first_day": part.index.min().date(),
                "persistence": mae_persistence,
                "trend_adjusted": mae_trend,
                "trend_minus_persistence": mae_trend - mae_persistence,
                "stronger": "persistence" if mae_persistence <= mae_trend else "trend-adjusted",
                # What the model must reach: 20 % below the stronger baseline.
                "target_mae": (1 - TARGET_GAIN) * min(mae_persistence, mae_trend),
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


# Runs main() only when the file is started as a script, not when it is imported.
if __name__ == "__main__":
    main()
