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
    # The target is 20 % below persistence on the same days.
    assert result.loc["test, all year", "target"] == pytest.approx(0.8 * result.loc["test, all year", "persistence"])
    # February to March: no October-to-January day.
    assert result.loc["test, Oct-Jan", "days"] == 0
