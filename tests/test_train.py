"""Tests for the training pipeline. They use small fake tables and no internet."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push.
import numpy as np
import pandas as pd
import pytest  # pytest.approx: compares decimal numbers with a tiny tolerance

from lakegeneva.common import baselines, features
from lakegeneva.training import train


def fake_rows(n_days: int, start: str = "2024-06-01", seed: int = 0) -> pd.DataFrame:
    """A fake feature table: random features, label = last reading + a known change."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n_days, freq="D", name="date")
    table = pd.DataFrame(rng.normal(0, 1, (n_days, len(features.FEATURES))), index=days, columns=features.FEATURES)
    table["water_last"] = 15 + rng.normal(0, 2, n_days)
    # Tomorrow changes by half of the south-west wind of tomorrow, nothing else.
    table[features.LABEL] = table["water_last"] + 0.5 * table["wind_sw_fc_d1"]
    return table


def test_a_model_for_day_d_learns_only_until_d_minus_2():
    rows = fake_rows(10)
    day = rows.index[7]

    known = train.known_at_18(rows, day)

    # The label of D-1 is the mean of D itself, which is not over at 18:00.
    assert known.index.max() == day - pd.Timedelta(days=2)


def test_labels_not_known_at_18_do_not_change_the_forecast(monkeypatch):
    # The main leakage test: the forecast for day D must not change if any label
    # that is unknown at 18:00 on D changes (the labels of D-1, D and later).
    monkeypatch.setattr(train, "MIN_TRAIN_ROWS", 10)
    rows = fake_rows(60)
    day = rows.index[40]
    before = train.walk_forward(rows, features.FEATURES, 1.0, [day])

    changed = rows.copy()
    changed.loc[changed.index >= day - pd.Timedelta(days=1), features.LABEL] += 100
    after = train.walk_forward(changed, features.FEATURES, 1.0, [day])

    assert after[day] == before[day]

    # Check that the test can notice a leak: a label that IS known (D-2) changes it.
    changed.loc[day - pd.Timedelta(days=2), features.LABEL] += 100
    assert train.walk_forward(changed, features.FEATURES, 1.0, [day])[day] != before[day]


def test_the_model_learns_a_simple_change(monkeypatch):
    monkeypatch.setattr(train, "MIN_TRAIN_ROWS", 30)
    rows = fake_rows(120)
    days = rows.index[60:]

    forecast = train.walk_forward(rows, features.FEATURES, 0.1, days)

    # The change is exactly half of one feature, so the model is almost perfect,
    # while persistence misses the whole change.
    assert train.mae(rows, forecast) < 0.05
    assert train.mae(rows, rows.loc[days, "water_last"]) > 0.3


def test_a_very_strong_penalty_gives_persistence_plus_the_mean_change(monkeypatch):
    monkeypatch.setattr(train, "MIN_TRAIN_ROWS", 10)
    rows = fake_rows(40)
    day = rows.index[-1]

    forecast = train.walk_forward(rows, features.FEATURES, 1e12, [day])

    # All coefficients are pushed to 0: only the average change of the training rows is left.
    known = train.known_at_18(rows, day)
    expected = rows.at[day, "water_last"] + train.change(known).mean()
    assert forecast[day] == pytest.approx(expected)


def test_days_with_too_few_earlier_rows_get_no_forecast(monkeypatch):
    monkeypatch.setattr(train, "MIN_TRAIN_ROWS", 10)
    rows = fake_rows(20)

    forecast = train.walk_forward(rows, features.FEATURES, 1.0, rows.index)

    # The first forecast needs 10 known rows: rows 0-9 are known on day 11 (index 11).
    assert forecast.index.min() == rows.index[11]


def test_a_window_uses_only_the_most_recent_rows(monkeypatch):
    monkeypatch.setattr(train, "MIN_TRAIN_ROWS", 10)
    rows = fake_rows(80)
    day = rows.index[-1]
    before = train.walk_forward(rows, features.FEATURES, 1.0, [day], window_days=30)

    # A row older than the 30-day window does not change the forecast ...
    changed = rows.copy()
    changed.loc[day - pd.Timedelta(days=40), features.LABEL] += 100
    assert train.walk_forward(changed, features.FEATURES, 1.0, [day], window_days=30)[day] == before[day]
    # ... a row inside the window does.
    changed.loc[day - pd.Timedelta(days=10), features.LABEL] += 100
    assert train.walk_forward(changed, features.FEATURES, 1.0, [day], window_days=30)[day] != before[day]


def test_a_kept_model_does_not_learn_from_newer_rows(monkeypatch):
    monkeypatch.setattr(train, "MIN_TRAIN_ROWS", 10)
    rows = fake_rows(80)
    days = rows.index[50:57]  # one week: a new model only on the first day
    weekly = train.walk_forward(rows, features.FEATURES, 1.0, days, retrain_every=7)

    # The label of row 52 becomes known on day 54 (2 days later). The weekly model was
    # trained on day 50, so its forecast for day 56 must not change; a daily model's does.
    changed = rows.copy()
    changed.loc[rows.index[52], features.LABEL] += 100
    assert train.walk_forward(changed, features.FEATURES, 1.0, days, retrain_every=7)[days[-1]] == weekly[days[-1]]
    daily = train.walk_forward(rows, features.FEATURES, 1.0, days)
    assert train.walk_forward(changed, features.FEATURES, 1.0, days)[days[-1]] != daily[days[-1]]


def test_a_gap_in_the_window_keeps_the_last_model(monkeypatch):
    monkeypatch.setattr(train, "MIN_TRAIN_ROWS", 10)
    rows = fake_rows(80)
    # A gap of 30 days in the data (rows 30 to 59 are missing).
    rows = rows.drop(rows.index[30:60])

    forecast = train.walk_forward(rows, features.FEATURES, 1.0, rows.index, window_days=20)

    # Right after the gap, the last 20 days hold almost no rows, so no new model can
    # be trained. The model of the last day before the gap is kept: every day from the
    # first model on (day 11, see the test above) has a forecast (found in a review on
    # 15.11.2026: before, the 11 days after the gap had none).
    assert list(forecast.index) == list(rows.index[11:])


def test_the_test_days_never_choose_the_setting(monkeypatch):
    monkeypatch.setattr(train, "MIN_TRAIN_ROWS", 20)
    monkeypatch.setattr(train, "ALPHAS", (0.1, 1000.0))
    # Rows on both sides of TEST_START (21.01.2025).
    rows = fake_rows(80, start="2024-12-01")
    alpha, errors = train.choose_alpha(rows, features.FEATURES)

    # Wild labels on the test days: the validation errors and the setting stay the same.
    changed = rows.copy()
    changed.loc[changed.index >= baselines.TEST_START, features.LABEL] = 1000.0
    assert train.choose_alpha(changed, features.FEATURES) == (alpha, errors)


def test_the_setting_uses_only_labels_known_at_the_first_test_forecast(monkeypatch):
    monkeypatch.setattr(train, "MIN_TRAIN_ROWS", 20)
    monkeypatch.setattr(train, "ALPHAS", (0.1, 1000.0))
    rows = fake_rows(80, start="2024-12-01")
    alpha, errors = train.choose_alpha(rows, features.FEATURES)

    # The label of the day before TEST_START is the mean of the first test day, which
    # is not over at 18:00 on that day: changing it must not change the setting.
    changed = rows.copy()
    changed.loc[baselines.TEST_START - pd.Timedelta(days=1), features.LABEL] += 100
    assert train.choose_alpha(changed, features.FEATURES) == (alpha, errors)

    # The label of TEST_START - 2 (the mean of the day before) is known: it counts.
    changed.loc[baselines.TEST_START - pd.Timedelta(days=2), features.LABEL] += 100
    assert train.choose_alpha(changed, features.FEATURES)[1] != errors


def test_choosing_alpha_stops_when_no_validation_day_can_be_scored():
    # A table that starts 11 days before TEST_START: no validation day has 90 earlier
    # rows, so every validation error is empty. Before the fix, choose_alpha quietly
    # returned the first alpha (0.1) (found in a review on 15.11.2026).
    rows = fake_rows(80, start="2025-01-10")

    # pytest.raises: the test passes only if this error comes, with this text in it.
    with pytest.raises(ValueError, match="alpha cannot be chosen"):
        train.choose_alpha(rows, features.FEATURES)


def test_model_rows_need_all_features_and_a_label():
    rows = fake_rows(3)
    rows.iloc[0, rows.columns.get_loc("air_temp_fc_d1")] = np.nan  # a missing forecast
    rows.iloc[1, rows.columns.get_loc(features.LABEL)] = np.nan  # tomorrow not over yet

    # Only the third day is complete. The water-only model uses the same rows, so
    # both models are compared on exactly the same days.
    assert list(train.model_rows(rows).index) == [rows.index[2]]


def test_evaluate_compares_with_persistence_in_each_window():
    rows = fake_rows(60, start="2025-02-01")
    # A perfect forecast: the error is 0, the gain 100 %.
    forecast = rows[features.LABEL].rename("forecast")

    result = train.evaluate(rows, forecast)

    assert list(result.index) == ["test, all year", "test, change > 0.8", "test, Oct-Jan"]
    assert result.loc["test, all year", "model"] == 0
    assert result.loc["test, all year", "gain_%"] == pytest.approx(100)
    # The target is 20 % below the target baseline (the frozen trend factor).
    assert result.loc["test, all year", "target"] == pytest.approx(0.8 * result.loc["test, all year", "trend_adjusted"])
    # February to March: no October-to-January day.
    assert result.loc["test, Oct-Jan", "days"] == 0
    # The big-change days also as events (runs of consecutive days, review point B3).
    big = baselines.big_change_days(rows.loc[forecast.index])
    assert result.loc["test, change > 0.8", "events"] == baselines.count_events(rows.index[big.to_numpy()])


def test_a_forecast_with_missing_values_gives_no_result():
    rows = fake_rows(60, start="2025-02-01")
    # Plain persistence as the "model", then its 8 worst days set to NaN (not a number).
    forecast = rows["water_last"].rename("forecast")
    worst = (forecast - rows[features.LABEL]).abs().nlargest(8).index
    forecast[worst] = np.nan

    # Before, mean() skipped the 8 days, the MAE got better and the table still said 60
    # days (review of 18.11.2026). Now there is no MAE and no target at all.
    with pytest.raises(ValueError, match="not a number"):
        train.evaluate(rows, forecast)


def test_every_expected_day_needs_a_forecast():
    days = pd.date_range("2025-02-01", periods=5, freq="D")
    forecast = pd.Series(1.0, index=days[:4])  # the last day has no forecast

    with pytest.raises(ValueError, match="1 of 5 days have no forecast"):
        train.check_every_day_forecast(forecast, days)
    # All five days with a number: no error.
    train.check_every_day_forecast(pd.Series(1.0, index=days), days)


def test_the_target_comes_from_the_frozen_baseline_even_when_persistence_wins():
    rows = fake_rows(60, start="2025-02-01")
    # A big 3-day trend: on these test days the target baseline (water_last - 0.075 x
    # trend) is far worse than plain persistence. Before (MS2 review point B1) the
    # target then switched to the better baseline on these very days, persistence.
    # Now the baseline was chosen once on the validation days and stays.
    rows["water_trend_3d"] = 10.0
    # The model forecast here is plain persistence itself.
    forecast = rows["water_last"].rename("forecast")

    result = train.evaluate(rows, forecast).loc["test, all year"]

    assert result["persistence"] < result["trend_adjusted"]
    assert result["target"] == pytest.approx(0.8 * result["trend_adjusted"])
    assert result["target"] != pytest.approx(0.8 * result["persistence"])
    # The gain and the interval are against the same target baseline.
    assert result["gain_%"] == pytest.approx(100 * (1 - result["persistence"] / result["trend_adjusted"]))
    assert result["diff"] == pytest.approx(result["persistence"] - result["trend_adjusted"])
