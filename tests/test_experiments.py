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


def test_a_lake_point_with_the_same_weather_scores_the_same(small, monkeypatch):
    rows, _ = small
    monkeypatch.setattr(train, "ALPHAS", (0.1, 100.0))
    # The "lake" has exactly Geneva's weather (review point B2): replacing the weather
    # features changes nothing, and the extra lake wind only adds a copy.
    lake = rows[features.WEATHER_FEATURES].copy()

    comparison, differences = experiments.lake_point_comparison(rows, lake)

    assert list(comparison.index) == [
        "Geneva (the features now)",
        "lake point instead of Geneva",
        "Geneva + the lake's south-west wind",
    ]
    geneva, instead = comparison.iloc[0], comparison.iloc[1]
    assert instead["validation MAE (decides)"] == pytest.approx(geneva["validation MAE (decides)"])
    assert instead["test MAE"] == pytest.approx(geneva["test MAE"])
    # The difference of "lake instead" to Geneva is 0 on every window that has days,
    # also on the validation year, which decides.
    rows_instead = differences[differences.index.str.startswith("lake point instead of Geneva")]
    assert "lake point instead of Geneva - Geneva, validation year, all days (decides)" in rows_instead.index
    assert rows_instead["diff"].dropna().abs().max() == pytest.approx(0)


def test_each_lake_version_gets_its_own_weather_on_the_same_days(small, monkeypatch):
    rows, _ = small
    monkeypatch.setattr(train, "ALPHAS", (0.1,))
    # Lake weather that differs from Geneva's in every value (random, so it also gives
    # other forecasts; a constant shift would not, the scaler removes it), and one
    # validation day without it (review of 26.11.2026: with the same weather at both
    # points, a version that quietly kept Geneva's weather passed every test).
    random = np.random.default_rng(1).normal(0, 1, (len(rows), len(features.WEATHER_FEATURES)))
    lake = pd.DataFrame(random, index=rows.index, columns=features.WEATHER_FEATURES)
    missing = rows.index[30]  # 31.12.2024
    lake.loc[missing] = np.nan
    # Remember the rows and columns that each version chooses its setting on.
    given = []
    real_choose_alpha = train.choose_alpha

    def remember(part, columns):
        given.append((part.copy(), list(columns)))
        return real_choose_alpha(part, columns)

    monkeypatch.setattr(train, "choose_alpha", remember)

    comparison, differences = experiments.lake_point_comparison(rows, lake)

    (geneva, geneva_columns), (instead, instead_columns), (added, added_columns) = given
    weather, water = features.WEATHER_FEATURES, features.WATER_FEATURES + [features.LABEL]
    kept = rows.drop(missing)
    # The same days for all three versions: without the day that has no lake weather.
    for part in (geneva, instead, added):
        assert part.index.equals(kept.index)
    assert comparison["test days"].nunique() == 1
    # Version 1: Geneva's weather. Version 2: the lake's weather, with the same water
    # features and labels. Version 3: Geneva's weather plus the lake's south-west wind.
    pd.testing.assert_frame_equal(geneva[weather], kept[weather])
    pd.testing.assert_frame_equal(instead[weather], lake.drop(missing))
    pd.testing.assert_frame_equal(instead[water], kept[water])
    pd.testing.assert_frame_equal(added[weather], kept[weather])
    extra = ["wind_sw_lake_d0", "wind_sw_lake_d1"]
    np.testing.assert_array_equal(
        added[extra].to_numpy(), lake.drop(missing)[["wind_sw_fc_d0", "wind_sw_fc_d1"]].to_numpy()
    )
    assert geneva_columns == instead_columns == train.FEATURES_FULL
    assert added_columns == train.FEATURES_FULL + extra
    # The deciding row compares the validation forecasts (not the test days): its
    # difference is the difference of the two validation MAEs, on all validation forecasts.
    deciding = differences.loc["lake point instead of Geneva - Geneva, validation year, all days (decides)"]
    validation_mae = comparison["validation MAE (decides)"]
    assert validation_mae.iloc[1] != pytest.approx(validation_mae.iloc[0])
    assert deciding["diff"] == pytest.approx(validation_mae.iloc[1] - validation_mae.iloc[0])
    on_validation = train.walk_forward(kept, train.FEATURES_FULL, 0.1, baselines.validation_days(kept).index)
    assert deciding["days"] == len(on_validation)
    # The big-change days of the validation year: their own MAE and difference, with
    # the events next to the day count (review of 26.11.2026).
    big = baselines.big_change_days(kept.loc[on_validation.index]).to_numpy()
    big_row = differences.loc["lake point instead of Geneva - Geneva, validation year, change > 0.8"]
    assert big_row["days"] == big.sum() > 0
    assert big_row["events"] == baselines.count_events(on_validation.index[big])
    big_mae = comparison["validation change > 0.8 MAE"]
    assert big_row["diff"] == pytest.approx(big_mae.iloc[1] - big_mae.iloc[0])
    geneva_errors = (on_validation - kept.loc[on_validation.index, features.LABEL]).abs()
    assert big_mae.iloc[0] == pytest.approx(geneva_errors[big].mean())
    # The test days are reported in the same way, as the difference of the test MAEs.
    on_test = differences.loc["lake point instead of Geneva - Geneva, test days, all days"]
    assert on_test["diff"] == pytest.approx(comparison["test MAE"].iloc[1] - comparison["test MAE"].iloc[0])


def test_the_lake_forecast_is_turned_into_the_same_weather_features(monkeypatch):
    # A fake answer of the Previous Runs API for two Swiss days (UTC hours, winter).
    hours = pd.date_range("2025-01-14 23:00", periods=48, freq="h", tz="UTC")
    answer = {"time": hours.strftime("%Y-%m-%dT%H:%M").tolist()}
    for column, value in ((features.TEMPERATURE, 4.0), (features.SUNSHINE, 1800.0), (features.WIND_SPEED, 10.0), (features.WIND_DIRECTION, 225.0)):
        answer[column] = [value] * 48
    asked = []

    def fake_fetch(start, end, latitude, longitude, cell_selection=None):
        asked.append((latitude, longitude, cell_selection))
        return answer

    monkeypatch.setattr(experiments.ingest_weather_runs, "fetch", fake_fetch)

    lake = experiments.lake_weather(pd.Timestamp("2025-01-15").date(), pd.Timestamp("2025-01-16").date())

    # The lake point was asked for, not Geneva, with the grid cell over the water ...
    assert asked == [(experiments.location.LAKE_LATITUDE, experiments.location.LAKE_LONGITUDE, "nearest")]
    # ... and the shared weather features come out: wind from the south-west, 10 km/h.
    assert lake.loc[pd.Timestamp("2025-01-15"), "air_temp_fc_d1"] == pytest.approx(4.0)
    assert lake.loc[pd.Timestamp("2025-01-15"), "wind_sw_fc_d0"] == pytest.approx(10.0)
