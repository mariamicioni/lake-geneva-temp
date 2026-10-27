"""Tests for the shared feature definition. They do not use the internet."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push
# (.github/workflows/ci.yml). pytest finds every function whose name starts with
# test_ and runs it; a test fails if one of its `assert` lines is false.
#
# Most tests build their own readings with known values, so the right answer can
# be worked out by hand. The last test uses a small sample of real readings
# (tests/fixtures/water_2606_sample.csv: 27.-30.03.2026, from the BAFU backfill).
from pathlib import Path

import pandas as pd
import pytest  # pytest.approx: compares decimal numbers with a tiny tolerance

from lakegeneva.common import features

FIXTURES = Path(__file__).parent / "fixtures"


def swiss_day(day: str, value: float = 10.0, flow: float = 100.0) -> pd.DataFrame:
    """10-minute readings for one whole Swiss day, all with the same value."""
    # From 00:00 of the day to 00:00 of the next day on the Swiss clock.
    # DateOffset(days=1) means "next calendar day", so a clock-change day gets its
    # 23 or 25 hours (138 or 150 readings), like the real data.
    start = pd.Timestamp(day, tz="Europe/Zurich")
    end = start + pd.DateOffset(days=1)
    times = pd.date_range(start, end, freq="10min", inclusive="left")  # "left": without the end
    return pd.DataFrame({"time_local": times, "water_temp_c": value, "flow_m3s": flow})


def test_cutoff_is_18_00_swiss_time_in_winter_and_summer():
    # One winter day (UTC+1) and one summer day (UTC+2).
    for day in ["2026-01-15", "2026-07-15"]:
        readings = swiss_day(day)
        # Each reading gets its Swiss clock time as value: 17:50 -> 17.5.
        readings["water_temp_c"] = readings["time_local"].dt.hour + readings["time_local"].dt.minute / 100

        row = features.water_features(readings).loc[pd.Timestamp(day)]

        # In both seasons the last reading before 18:00 is the one of 17:50 on the
        # Swiss clock, and 108 readings (00:00 to 17:50) count for today.
        assert row["water_last"] == 17.5
        assert row["n_readings_d0"] == 108


def test_readings_after_18_00_do_not_change_the_features():
    # The leakage test. Five whole days, 10.-14.10.2026, with falling temperature.
    readings = pd.concat([swiss_day(f"2026-10-{d}", value=20 - d) for d in range(10, 15)], ignore_index=True)
    before = features.water_features(readings)

    # Now set every reading after 18:00 on 13.10 to an impossible 99 °C.
    changed = readings.copy()
    late = (changed["time_local"].dt.day == 13) & (changed["time_local"].dt.hour >= 18)
    changed.loc[late, "water_temp_c"] = 99.0
    after = features.water_features(changed)

    # The features of 13.10 are exactly the same: nothing after 18:00 leaks in.
    # (The 3-day mean of 14.10 may change: at 18:00 on 14.10 all of 13.10 is past.)
    day = pd.Timestamp("2026-10-13")
    pd.testing.assert_series_equal(before.loc[day, features.WATER_FEATURES], after.loc[day, features.WATER_FEATURES])
    # Check that the change was real: the label of 12.10 (the mean of all of 13.10) moved.
    previous_day = pd.Timestamp("2026-10-12")
    assert after.loc[previous_day, features.LABEL] != before.loc[previous_day, features.LABEL]


def test_label_needs_at_least_120_readings_of_tomorrow():
    today = swiss_day("2026-10-10", value=15.0)
    tomorrow = swiss_day("2026-10-11", value=14.0)
    # iloc[:n] keeps only the first n readings of tomorrow.
    few = features.water_features(pd.concat([today, tomorrow.iloc[:119]], ignore_index=True))
    enough = features.water_features(pd.concat([today, tomorrow.iloc[:120]], ignore_index=True))

    day = pd.Timestamp("2026-10-10")
    assert pd.isna(few.loc[day, features.LABEL])  # 119 readings: no label
    assert enough.loc[day, features.LABEL] == 14.0  # 120 readings: the mean of tomorrow


def test_three_day_mean_and_trend():
    # Four whole days at 10, 11, 12 and 13 °C.
    days = [("2026-10-10", 10.0), ("2026-10-11", 11.0), ("2026-10-12", 12.0), ("2026-10-13", 13.0)]
    readings = pd.concat([swiss_day(day, value) for day, value in days], ignore_index=True)

    row = features.water_features(readings).loc[pd.Timestamp("2026-10-13")]

    # 3-day mean on 13.10: all 144 readings of 11.10 and of 12.10, plus the 108
    # readings of 13.10 before 18:00.
    assert row["water_mean_3d"] == pytest.approx((144 * 11 + 144 * 12 + 108 * 13) / (144 + 144 + 108))
    # Trend: today's 00:00-18:00 mean (13) minus the same hours three days earlier (10).
    assert row["water_trend_3d"] == pytest.approx(3.0)


def test_day_with_too_few_readings_before_18_00_gets_no_features():
    # Only the first 89 readings of the day: one less than the rule's 90.
    readings = swiss_day("2026-10-10").iloc[:89]

    row = features.water_features(readings).loc[pd.Timestamp("2026-10-10")]

    assert row["n_readings_d0"] == 89
    assert row[features.WATER_FEATURES].isna().all()  # every feature is empty


def test_clock_change_day_counts_as_a_whole_day():
    # 29.03.2026: the clocks jump from 02:00 to 03:00, so this Swiss day has only
    # 23 hours = 138 readings. It must still give a label (138 >= 120).
    readings = pd.concat([swiss_day("2026-03-28"), swiss_day("2026-03-29", value=12.0)], ignore_index=True)

    row = features.water_features(readings).loc[pd.Timestamp("2026-03-28")]

    assert row["n_readings_d1"] == 138
    assert row[features.LABEL] == 12.0


def test_load_water_reads_all_files_and_keeps_the_newest_duplicate(tmp_path):
    # Two raw files with one shared reading time, like the daily download that
    # overlaps the backfill. tmp_path (a pytest helper) is a fresh temporary folder.
    (tmp_path / "water_2606_10min_2026-10-01_to_2026-10-01.csv").write_text(
        "time_utc,water_temp_c,flow_m3s\n"
        "2026-10-01T10:00:00+00:00,18.50,200.0\n"
        "2026-10-01T10:10:00+00:00,18.40,200.0\n"
    )
    (tmp_path / "water_2606_fetched_2026-10-02.csv").write_text(
        "time_utc,water_temp_c,flow_m3s\n2026-10-01T10:10:00+00:00,18.45,201.0\n"
    )

    readings = features.load_water(tmp_path)

    # Two different times; the shared one comes from the daily file (later in name order).
    assert list(readings["water_temp_c"]) == [18.50, 18.45]
    # The times are now Swiss time: 10:00 UTC is 12:00 in summer time.
    assert readings["time_local"].iloc[0] == pd.Timestamp("2026-10-01 12:00", tz="Europe/Zurich")


def test_real_sample_around_the_clock_change():
    # Real BAFU readings of 27.-30.03.2026, read the same way as the full data.
    table = features.water_features(features.load_water(FIXTURES))

    # One row per Swiss day.
    assert list(table.index.strftime("%Y-%m-%d")) == ["2026-03-27", "2026-03-28", "2026-03-29", "2026-03-30"]
    # Readings of tomorrow: 144 for 28.03, 138 for the clock-change day 29.03, 144
    # for 30.03, and none after the sample ends.
    assert list(table["n_readings_d1"]) == [144, 138, 144, 0]
    # Every day of the real sample passes the 18:00 rule, so water_last always exists.
    assert table["water_last"].notna().all()
