"""Tests for the two baselines and the bootstrap. They do not use the internet."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push
# (.github/workflows/ci.yml). pytest finds every function whose name starts with
# test_ and runs it; a test fails if one of its `assert` lines is false.
import numpy as np
import pandas as pd
import pytest  # pytest.approx: compares decimal numbers with a tiny tolerance

from lakegeneva.common import baselines, features


def feature_rows(n_days: int, start: str = "2025-10-01", water_last=10.0, trend=0.3, label=10.5) -> pd.DataFrame:
    """A fake feature table: n days, every feature filled, known baseline errors."""
    table = pd.DataFrame(
        {column: 1.0 for column in features.FEATURES},  # every feature exists (value does not matter)
        index=pd.date_range(start, periods=n_days, freq="D", name="date"),
    )
    table["water_last"] = water_last
    table["water_trend_3d"] = trend
    table[features.LABEL] = label
    return table


def test_persistence_and_trend_adjusted():
    table = feature_rows(1, water_last=10.0, trend=3.0)
    # Persistence: the last reading. Trend-adjusted: plus a third of the 3-day trend.
    assert baselines.persistence(table).iloc[0] == 10.0
    assert baselines.trend_adjusted(table).iloc[0] == pytest.approx(11.0)


def test_big_change_days_are_those_with_more_than_0_8_degrees():
    table = feature_rows(3, water_last=10.0)
    table[features.LABEL] = [10.81, 10.79, 9.1]  # +0.81, +0.79 and -0.9 °C
    assert list(baselines.big_change_days(table)) == [True, False, True]


def test_bootstrap_of_a_constant_difference_has_no_spread():
    # a is always exactly 0.1 worse than b: every new test set gives 0.1.
    errors_b = np.random.default_rng(1).uniform(0, 1, 100)
    diff, low, high = baselines.bootstrap_mae_difference(errors_b + 0.1, errors_b)
    assert diff == pytest.approx(0.1)
    assert low == pytest.approx(0.1)
    assert high == pytest.approx(0.1)


def test_bootstrap_is_reproducible_and_contains_the_difference():
    rng = np.random.default_rng(2)
    errors_a, errors_b = rng.uniform(0, 1, 200), rng.uniform(0, 1, 200)

    first = baselines.bootstrap_mae_difference(errors_a, errors_b)
    second = baselines.bootstrap_mae_difference(errors_a, errors_b)

    # The fixed seed gives exactly the same interval on every run ...
    assert first == second
    # ... and the measured difference lies inside its own interval.
    diff, low, high = first
    assert low <= diff <= high


def test_bootstrap_handles_a_long_gap_without_days():
    # Two parts of the year with very different differences: +1 in January, -1 in
    # July, and no day in between.
    january = pd.date_range("2025-01-01", periods=40, freq="D")
    july = pd.date_range("2025-07-01", periods=40, freq="D")
    dates = january.append(july)
    errors_b = pd.Series(1.0, index=dates)
    errors_a = errors_b + pd.Series([1.0] * 40 + [-1.0] * 40, index=dates)

    diff, low, high = baselines.bootstrap_mae_difference(errors_a, errors_b)

    assert diff == pytest.approx(0)
    # Most weeks between August and December hold no day of the part. Such empty weeks
    # add nothing to a new test set, so the interval is still a normal, finite one.
    assert np.isfinite(low) and np.isfinite(high)
    assert low < 0 < high


def test_bootstrap_with_dates_gives_the_same_as_plain_numbers_on_consecutive_days():
    rng = np.random.default_rng(3)
    errors_a, errors_b = rng.uniform(0, 1, 120), rng.uniform(0, 1, 120)
    dates = pd.date_range("2025-03-01", periods=120, freq="D")

    plain = baselines.bootstrap_mae_difference(errors_a, errors_b)
    with_dates = baselines.bootstrap_mae_difference(pd.Series(errors_a, index=dates), pd.Series(errors_b, index=dates))

    # Without dates the numbers count as consecutive days, so both give the same result.
    assert plain == pytest.approx(with_dates)


def test_bootstrap_blocks_are_weeks_of_the_calendar_not_rows():
    # Six days of a part, one per week, with differences 1, 2, ... 6 (like big-change
    # days, which are far apart).
    dates = pd.date_range("2025-03-03", periods=6, freq="7D")
    errors_b = pd.Series(0.0, index=dates)
    errors_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=dates)

    diff, low, high = baselines.bootstrap_mae_difference(errors_a, errors_b)

    assert diff == pytest.approx(3.5)
    # A block of 7 calendar days holds one of these days, so each new test set draws
    # single days and the result spreads. Blocks of 7 ROWS would take all six days
    # every time: an interval of width 0 that looks far too sure (the MS2 problem).
    assert 1.0 <= low < diff < high <= 6.0
    assert high - low > 1.0


def test_report_compares_both_baselines_on_the_same_days():
    # 60 October/November days: persistence is 0.5 off every day (10.0 vs 10.5),
    # trend-adjusted 0.4 (10.1 vs 10.5), so here trend-adjusted is the stronger one.
    result = baselines.report(feature_rows(60))

    row = result.loc["all features, all year"]
    assert row["days"] == 60
    assert row["persistence"] == pytest.approx(0.5)
    assert row["trend_adjusted"] == pytest.approx(0.4)
    assert row["stronger"] == "trend-adjusted"
    # The target is 20 % below the STRONGER baseline: 0.8 x 0.4.
    assert row["target_mae"] == pytest.approx(0.32)
    # All 60 days are in October-January, and none changes by more than 0.8 °C.
    assert result.loc["all features, Oct-Jan", "days"] == 60
    assert result.loc["all features, change > 0.8", "days"] == 0


def test_report_counts_only_the_test_days_from_21_01_2025():
    # 20 days from 11.01.2025: 10 days of the validation year (11.-20.01.) and
    # 10 test days (21.-30.01.). Only the test days count for the target.
    result = baselines.report(feature_rows(20, start="2025-01-11"))

    assert result.loc["test, all year", "days"] == 10
    assert result.loc["test, all year", "first_day"] == pd.Timestamp("2025-01-21").date()
    # The windows over all days with all features still see all 20 days.
    assert result.loc["all features, all year", "days"] == 20


def test_report_gives_no_interval_for_a_handful_of_days():
    # 10 of the 60 days change by 1 °C: too few days (under 35) for an interval.
    table = feature_rows(60)
    table.iloc[:10, table.columns.get_loc(features.LABEL)] = 11.0

    row = baselines.report(table).loc["all features, change > 0.8"]

    assert row["days"] == 10
    assert row["persistence"] == pytest.approx(1.0)  # the MAE is still shown ...
    assert pd.isna(row["ci_low"]) and pd.isna(row["ci_high"])  # ... but no interval
