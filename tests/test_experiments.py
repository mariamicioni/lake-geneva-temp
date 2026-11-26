"""Tests for the training experiments (ladder, data, retrain). Small fake tables, no internet."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push.
import numpy as np  # np.nan: a forecast that is not a number
import pandas as pd
import pytest

from lakegeneva.common import baselines, features
from lakegeneva.training import experiments, train
from tests.test_train import fake_rows


@pytest.fixture
def small(monkeypatch):
    """A fake table across TEST_START and settings small enough for a fast test."""
    monkeypatch.setattr(train, "MIN_TRAIN_ROWS", 20)
    monkeypatch.setattr(experiments, "WINDOWS", {"last 30 days": 30, "all earlier rows": None})
    monkeypatch.setattr(experiments, "TIMING_REPEATS", 1)
    rows = fake_rows(120, start="2024-12-01")
    test_days = rows.index[rows.index >= baselines.TEST_START]
    return rows, test_days


def test_ladder_has_every_rung_on_the_same_days(small):
    rows, test_days = small

    result = experiments.ladder(rows, test_days, {"full": 0.1, "water_only": 0.1})

    assert list(result.index) == [
        "1 persistence",
        "2 trend-adjusted persistence",
        "3 persistence + average change",
        "4 Ridge, water only",
        "5 Ridge, all features",
    ]
    # In the fake data the change depends only on a weather feature: the full model
    # learns it, persistence and the water-only model cannot.
    assert result.loc["5 Ridge, all features", "test MAE"] < result.loc["4 Ridge, water only", "test MAE"]
    assert result.loc["5 Ridge, all features", "test MAE"] < result.loc["1 persistence", "test MAE"]
    # The baselines need no training; the models report a time.
    assert result.loc["1 persistence", "one fit (ms)"] == 0
    assert result.loc["3 persistence + average change", "one fit (ms)"] == 0
    assert result.loc["5 Ridge, all features", "one fit (ms)"] > 0


def test_average_change_corrects_a_constant_offset():
    rows = fake_rows(30)
    # Tomorrow's mean is always 0.3 degC below the last reading (an afternoon reading is warm).
    rows[features.LABEL] = rows["water_last"] - 0.3
    days = rows.index[10:]

    forecast = experiments.average_change(rows, days)

    # One number corrects it completely, while plain persistence is always 0.3 off.
    assert train.mae(rows, forecast) == pytest.approx(0)
    assert train.mae(rows, rows.loc[days, "water_last"]) == pytest.approx(0.3)
    # Only rows known at 18:00 count: a wild label on the day before changes nothing.
    changed = rows.copy()
    changed.loc[days[-1] - pd.Timedelta(days=1), features.LABEL] += 100
    assert experiments.average_change(changed, [days[-1]])[days[-1]] == forecast[days[-1]]


def test_forecast_comparison_uses_only_days_with_both_forecasts(small):
    rows, _ = small
    # The 06:00 UTC run gives exactly the same weather here, except that it is missing
    # on the first 10 days (like before 10.08.2024 in the real data).
    single = rows[features.WEATHER_FEATURES].copy()
    single.iloc[:10] = None

    result, differences = experiments.forecast_comparison(rows, single)

    first, second = result.index
    # The same days for both, and with the same features the same error.
    assert result.loc[first, "test days"] == result.loc[second, "test days"]
    assert result.loc[first, "test MAE"] == pytest.approx(result.loc[second, "test MAE"])
    # So the difference is 0, and its interval is exactly 0 too.
    assert differences.loc["all test days", "days"] == result.loc[first, "test days"]
    assert differences.loc["all test days", "diff"] == pytest.approx(0)
    assert differences.loc["all test days", "ci_low"] == pytest.approx(0)
    assert differences.loc["all test days", "ci_high"] == pytest.approx(0)


def test_intervals_compare_the_rungs_of_the_ladder(small):
    rows, test_days = small
    alphas = {"full": 0.1, "water_only": 0.1}

    result = experiments.intervals(rows, test_days, alphas)
    ladder = experiments.ladder(rows, test_days, alphas)

    # Five comparisons, each in the three windows of the README.
    assert len(result) == 5 * 3
    # The model against the target baseline: the same difference as in the ladder.
    full_vs_target = result.loc[("5 all features - 2 target baseline", "all test days")]
    assert full_vs_target["diff"] == pytest.approx(
        ladder.loc["5 Ridge, all features", "test MAE"] - ladder.loc["2 trend-adjusted persistence", "test MAE"]
    )
    # The difference is the difference of the two MAEs in the ladder (the same days).
    full_vs_water = result.loc[("5 all features - 4 water only", "all test days")]
    expected = ladder.loc["5 Ridge, all features", "test MAE"] - ladder.loc["4 Ridge, water only", "test MAE"]
    assert full_vs_water["diff"] == pytest.approx(expected)
    # In the fake data only a weather feature matters: the full model is clearly
    # better, so the whole interval is below 0.
    assert full_vs_water["ci_high"] < 0
    # Only 10 test days in October to January here: a difference, but no interval
    # (fewer than 35 days would look far too sure).
    oct_jan = result.loc[("5 all features - 4 water only", "Oct-Jan")]
    assert oct_jan["days"] == 10
    assert pd.isna(oct_jan["ci_low"])
    # Next to the big-change days: how many events they are (review point B3), and only there.
    big = baselines.big_change_days(rows.loc[test_days]).to_numpy()
    big_row = result.loc[("5 all features - 4 water only", "change > 0.8")]
    assert big_row["days"] == big.sum()
    assert big_row["events"] == baselines.count_events(test_days[big])
    assert pd.isna(result.loc[("5 all features - 4 water only", "all test days"), "events"])


def test_a_missing_forecast_stops_the_experiments(small, monkeypatch):
    rows, test_days = small
    # The walk-forward test gives one forecast that is not a number (NaN).
    real_walk_forward = train.walk_forward

    def one_forecast_missing(*args, **kwargs):
        forecast = real_walk_forward(*args, **kwargs)
        forecast.iloc[5] = np.nan
        return forecast

    monkeypatch.setattr(train, "walk_forward", one_forecast_missing)

    # Before, mean() skipped that day and the README tables would have been computed on
    # fewer days without a word (review of 20.11.2026). Now every table stops.
    with pytest.raises(ValueError, match="not a number"):
        experiments.ladder(rows, test_days, {"full": 0.1, "water_only": 0.1})
    with pytest.raises(ValueError, match="not a number"):
        experiments.intervals(rows, test_days, {"full": 0.1, "water_only": 0.1})
    with pytest.raises(ValueError, match="not a number"):
        experiments.retrain_replay(rows, test_days, 0.1)


def test_learning_curve_and_retrain_cover_every_setting(small):
    rows, test_days = small

    curve = experiments.learning_curve(rows, test_days, 0.1)
    replay = experiments.retrain_replay(rows, test_days, 0.1)

    assert list(curve.index) == list(experiments.WINDOWS)
    assert list(replay.index) == list(experiments.RETRAIN)
    # A new model every day is the walk-forward test of train.py itself.
    daily = train.walk_forward(rows, train.FEATURES_FULL, 0.1, test_days)
    assert replay.loc["every day", "test MAE"] == pytest.approx(train.mae(rows, daily))
