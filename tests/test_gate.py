"""Tests for the promotion gate (gate.py). Small fake tables and fake numbers, no internet."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push.
import datetime as dt
import os  # os.linesep: the line end of the computer ("\n" on Linux, "\r\n" on Windows)

import numpy as np
import pandas as pd
import pytest  # pytest.fixture: a fresh fake table for every test

from lakegeneva.common import baselines, features
from lakegeneva.training import gate, train
from tests.test_train import fake_rows


@pytest.fixture
def rows(monkeypatch):
    """A fake table across TEST_START (21.01.2025), with a small MIN_TRAIN_ROWS."""
    monkeypatch.setattr(train, "MIN_TRAIN_ROWS", 20)
    return fake_rows(100, start="2024-12-01")


def good_summary(**changes) -> dict:
    """A validation summary of a healthy procedure; changes replace single numbers."""
    summary = {
        "validation_days": 274,
        "validation_md5": "abc",
        "validation_mae": 0.576,
        "validation_mae_persistence": 0.606,
        "validation_mae_row3": 0.595,
    }
    return summary | changes  # | : the same dict with the changed entries


def forecasts(values, start="2026-11-10") -> pd.Series:
    """Fake forecasts on consecutive days."""
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"), dtype=float)


# --- The most important rule: no label of a test day ------------------------------


def test_the_validation_summary_reads_no_label_of_a_test_day(rows):
    summary = gate.validation_summary(rows, features.FEATURES, 1.0)

    # Wild labels from the day before TEST_START on: the label of 20.01.2025 is the mean
    # of the first test day, and every later label belongs to the test period. The gate
    # must not see any of them: the whole summary stays exactly the same.
    changed = rows.copy()
    changed.loc[changed.index >= baselines.TEST_START - pd.Timedelta(days=1), features.LABEL] = 1000.0
    assert gate.validation_summary(changed, features.FEATURES, 1.0) == summary

    # Check that the test can notice: a validation label (known at the first test day) does change it.
    changed.loc[baselines.TEST_START - pd.Timedelta(days=10), features.LABEL] += 1
    assert gate.validation_summary(changed, features.FEATURES, 1.0) != summary


def test_the_fingerprint_follows_the_labels_but_not_the_features(rows):
    summary = gate.validation_summary(rows, features.FEATURES, 1.0)
    day = baselines.TEST_START - pd.Timedelta(days=5)  # a validation day with a forecast

    # Another label on a validation day: the MAEs are no longer comparable (other fingerprint).
    other_label = rows.copy()
    other_label.loc[day, features.LABEL] += 0.5
    assert gate.validation_summary(other_label, features.FEATURES, 1.0)["validation_md5"] != summary["validation_md5"]

    # Another feature (e.g. a bug in features.py): still comparable, so that G2 can find it.
    other_feature = rows.copy()
    other_feature.loc[day, "air_temp_fc_d1"] += 5
    assert gate.validation_summary(other_feature, features.FEATURES, 1.0)["validation_md5"] == summary["validation_md5"]


def test_the_fingerprint_is_the_same_on_windows_and_linux(rows, monkeypatch):
    # The same validation days on a laptop (Windows) and on GitHub (Linux) must give the
    # same fingerprint, or G2 could never compare the two (review of 18.11.2026).
    fingerprints = {}
    for computer, line_end in (("Linux", "\n"), ("Windows", "\r\n")):
        monkeypatch.setattr(os, "linesep", line_end)
        fingerprints[computer] = gate.validation_summary(rows, features.FEATURES, 1.0)["validation_md5"]

    assert fingerprints["Linux"] == fingerprints["Windows"]


def test_the_validation_days_end_before_the_test_period(rows):
    summary = gate.validation_summary(rows, features.FEATURES, 1.0)

    # The last validation day is 19.01.2025: its label (the mean of 20.01) is known at
    # 18:00 on the first test day. 20.01.2025 is neither a validation nor a test day.
    assert summary["validation_last"] == "2025-01-19"
    assert summary["validation_days"] > 0


# --- G1: better than the free forecasts ---------------------------------------------


def test_g1_needs_a_lower_mae_than_persistence_and_row_3(rows):
    # The fake procedure learns the change exactly, so it beats both free forecasts.
    passed, text = gate.check_free_forecasts(gate.validation_summary(rows, features.FEATURES, 1.0))
    assert passed, text

    # Worse than row 3 (but better than persistence): no.
    assert not gate.check_free_forecasts(good_summary(validation_mae=0.600))[0]
    # Worse than persistence: no.
    assert not gate.check_free_forecasts(good_summary(validation_mae=0.700))[0]


def test_one_missing_validation_forecast_makes_g1_and_g2_say_no(rows, monkeypatch):
    # The walk-forward test gives one forecast that is not a number (NaN).
    real_walk_forward = train.walk_forward

    def one_forecast_missing(*args, **kwargs):
        forecast = real_walk_forward(*args, **kwargs)
        forecast.iloc[3] = np.nan
        return forecast

    monkeypatch.setattr(train, "walk_forward", one_forecast_missing)
    summary = gate.validation_summary(rows, features.FEATURES, 1.0)

    # The MAE is not computed on the other days only: it is "not a number", so both
    # checks say no (found in a review on 17.11.2026: before, the day was skipped).
    assert np.isnan(summary["validation_mae"])
    assert not gate.check_free_forecasts(summary)[0]
    same_days = {"params": {"validation_md5": summary["validation_md5"]}, "metrics": {"validation_mae": 0.5}}
    assert not gate.check_champion_validation(summary, same_days)[0]


def test_g1_says_no_when_a_number_is_missing():
    # Fail-closed: NaN, infinity or no validation day at all is never a pass.
    assert not gate.check_free_forecasts(good_summary(validation_mae=float("nan")))[0]
    assert not gate.check_free_forecasts(good_summary(validation_mae_row3=float("inf")))[0]
    assert not gate.check_free_forecasts(good_summary(validation_days=0))[0]


# --- G2: not worse than the champion --------------------------------------------------


def champion(md5="abc", mae=0.576) -> dict:
    """What track.champion_info() gives for the champion's run (without the model)."""
    return {"params": {"validation_md5": md5}, "metrics": {"validation_mae": mae}}


def test_g2_allows_a_tiny_increase_only():
    # 0.004 worse than the champion: inside the 0.005 allowed.
    assert gate.check_champion_validation(good_summary(validation_mae=0.580), champion(mae=0.576))[0]
    # 0.006 worse: no.
    assert not gate.check_champion_validation(good_summary(validation_mae=0.582), champion(mae=0.576))[0]
    # Better than the champion: yes.
    assert gate.check_champion_validation(good_summary(validation_mae=0.570), champion(mae=0.576))[0]


def test_g2_does_not_compare_different_validation_days():
    # Other validation days or labels (another fingerprint): the two MAEs cannot be
    # compared, so the answer is no, and a person decides (track.promote_by_hand).
    passed, text = gate.check_champion_validation(good_summary(), champion(md5="other"))
    assert not passed and "not comparable" in text
    # A champion from before the gate has no fingerprint at all: also not comparable.
    passed, text = gate.check_champion_validation(good_summary(), {"params": {}, "metrics": {}})
    assert not passed and "not comparable" in text
    # The same fingerprint, but the champion's MAE is missing: no.
    assert not gate.check_champion_validation(good_summary(), {"params": {"validation_md5": "abc"}, "metrics": {}})[0]


def test_g2_is_skipped_only_without_a_champion():
    passed, text = gate.check_champion_validation(good_summary(), None)
    assert passed and text.startswith("skipped")


# --- G3: shadow check against the champion --------------------------------------------


def test_g3_allows_small_differences_only():
    new = forecasts([10.0, 10.5, 11.0])
    # 0.2 degC apart on one day: allowed (limit 0.3).
    assert gate.check_shadow(new, new + [0.0, 0.2, 0.0])[0]
    # 1 degC apart on one day: no.
    assert not gate.check_shadow(new, new + [0.0, 1.0, 0.0])[0]
    # No champion: skipped.
    assert gate.check_shadow(new, None)[1].startswith("skipped")


def test_g3_says_no_when_the_forecasts_cannot_be_compared():
    new = forecasts([10.0, 10.5])
    assert not gate.check_shadow(new, forecasts([10.0, np.nan]))[0]  # a NaN forecast
    assert not gate.check_shadow(new, forecasts([10.0, 10.5], start="2026-01-01"))[0]  # other days
    assert not gate.check_shadow(forecasts([]), forecasts([]))[0]  # no day at all


# --- G4: a plausible change -----------------------------------------------------------


def test_g4_rejects_a_change_bigger_than_ever_before_the_test_period():
    newest = pd.DataFrame({"water_last": [10.0, 12.0]}, index=forecasts([0, 0]).index)
    # Changes of -0.3 and +0.5 degC: plausible.
    assert gate.check_plausible(forecasts([9.7, 12.5]), newest)[0]
    # A change of +10 degC (e.g. a model that returns a temperature instead of a change): no.
    assert not gate.check_plausible(forecasts([20.0, 12.0]), newest)[0]
    # A change of -6 degC: the limit holds in both directions.
    assert not gate.check_plausible(forecasts([4.0, 12.0]), newest)[0]
    # An infinite forecast: no.
    assert not gate.check_plausible(forecasts([np.inf, 12.0]), newest)[0]


# --- G5: fresh data -------------------------------------------------------------------


def test_g5_needs_fresh_features_and_a_fresh_label():
    newest = pd.DataFrame({"water_last": [10.0, 10.1]}, index=pd.to_datetime(["2026-11-15", "2026-11-16"]))
    monday = dt.date(2026, 11, 16)

    # A normal Monday morning: features of Sunday (0-1 days), newest label Friday (3 days).
    assert gate.check_fresh_data(pd.Timestamp("2026-11-13"), newest, monday)[0]
    # Fresh features, but no new label for 37 days (the case of the review of 18.11.2026): no.
    passed, text = gate.check_fresh_data(pd.Timestamp("2026-10-10"), newest, monday)
    assert not passed and "label" in text
    # Features older than 2 days (the rule of check_fresh.py): no.
    assert not gate.check_fresh_data(pd.Timestamp("2026-11-13"), newest, dt.date(2026, 11, 20))[0]
    # No row with all features at all: no.
    assert not gate.check_fresh_data(pd.Timestamp("2026-11-13"), newest.iloc[:0], monday)[0]


# --- The newest rows and the final decision -------------------------------------------


def test_the_newest_rows_keep_the_days_without_a_label():
    table = fake_rows(20)
    table.iloc[-2:, table.columns.get_loc(features.LABEL)] = np.nan  # tomorrow is not over yet
    table.iloc[-5, table.columns.get_loc("air_temp_fc_d1")] = np.nan  # one forecast missing

    newest = gate.newest_rows(table)

    # The two days without a label are in (they are the days of the live forecast) ...
    assert newest.index[-1] == table.index[-1]
    assert newest[features.LABEL].isna().sum() == 2
    # ... the day without all features is not, and there are NEWEST_DAYS rows.
    assert table.index[-5] not in newest.index
    assert len(newest) == gate.NEWEST_DAYS


def test_a_model_is_promoted_only_if_every_check_passed_or_was_skipped():
    passed = {name: "passed: fine" for name in gate.CHECKS}
    assert gate.passes(passed)
    # Skipped (no champion yet) is allowed.
    assert gate.passes(passed | {"g3_shadow_vs_champion": "skipped: no champion yet"})
    # One failed check is enough for a no.
    assert not gate.passes(passed | {"g2_validation_vs_champion": "failed: not comparable"})
    # A missing check counts as no: a forgotten check can never promote a model.
    assert not gate.passes({name: text for name, text in passed.items() if name != "g5_fresh_data"})
    assert not gate.passes({})
