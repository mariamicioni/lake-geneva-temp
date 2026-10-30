"""Tests for the shared feature definition. They do not use the internet."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push
# (.github/workflows/ci.yml). pytest finds every function whose name starts with
# test_ and runs it; a test fails if one of its `assert` lines is false.
#
# Most tests build their own readings and forecasts with known values, so the right
# answer can be worked out by hand. Two tests use a small sample of real readings
# (tests/fixtures/water_2606_sample.csv: 27.-30.03.2026, from the BAFU backfill).
from pathlib import Path

import numpy as np  # cosine, to work out the expected wind value by hand
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
    # iloc[-n:] keeps only the last n readings of tomorrow: a gap in the night, but
    # the day is over (its 23:50 reading exists).
    few = features.water_features(pd.concat([today, tomorrow.iloc[-119:]], ignore_index=True))
    enough = features.water_features(pd.concat([today, tomorrow.iloc[-120:]], ignore_index=True))

    day = pd.Timestamp("2026-10-10")
    assert pd.isna(few.loc[day, features.LABEL])  # 119 readings: no label
    assert enough.loc[day, features.LABEL] == 14.0  # 120 readings: the mean of tomorrow


def test_label_waits_until_tomorrow_is_over():
    # Like the job of 27.10.2026 that started at 23:52: tomorrow already has 132
    # readings (00:00 to 21:50), more than 120, but the day is not over yet.
    today = swiss_day("2026-10-10", value=15.0)
    tomorrow = swiss_day("2026-10-11", value=14.0)
    not_over = features.water_features(pd.concat([today, tomorrow.iloc[:132]], ignore_index=True))
    # The next evening the whole day is there.
    over = features.water_features(pd.concat([today, tomorrow], ignore_index=True))

    day = pd.Timestamp("2026-10-10")
    assert pd.isna(not_over.loc[day, features.LABEL])  # not over: no label yet
    assert over.loc[day, features.LABEL] == 14.0  # over: the mean of all of 11.10


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


def test_load_water_does_not_lose_a_value_when_the_newer_file_has_a_gap(tmp_path):
    # The same time in two files. The newer file has the temperature but no flow.
    (tmp_path / "water_2606_10min_2026-10-01_to_2026-10-01.csv").write_text(
        "time_utc,water_temp_c,flow_m3s\n2026-10-01T10:00:00+00:00,18.50,200.0\n"
    )
    (tmp_path / "water_2606_fetched_2026-10-02.csv").write_text(
        "time_utc,water_temp_c,flow_m3s\n2026-10-01T10:00:00+00:00,18.45,\n"
    )

    readings = features.load_water(tmp_path)

    # One row: the newest temperature, and the flow from the older file.
    assert len(readings) == 1
    assert readings["water_temp_c"].iloc[0] == 18.45
    assert readings["flow_m3s"].iloc[0] == 200.0


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


# --- Weather part -------------------------------------------------------------------


def forecast_hours(day: str, temperature=10.0, sunshine=1800.0, speed=10.0, direction=225.0) -> pd.DataFrame:
    """All hours of a fake 48-hour-old forecast for one Swiss day, as load_weather() returns them."""
    # From 00:00 to 00:00 of the next day on the Swiss clock: 24 hours, or 23/25 on
    # a clock-change day (DateOffset(days=1) = the next calendar day).
    start = pd.Timestamp(day, tz="Europe/Zurich")
    local = pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="left")
    return pd.DataFrame(
        {
            "fetched_at_utc": pd.Timestamp("2026-10-27 19:00", tz="UTC"),
            "time_utc": local.tz_convert("UTC"),
            "time_local": local,
            features.TEMPERATURE: temperature,
            features.SUNSHINE: sunshine,
            features.WIND_SPEED: speed,
            features.WIND_DIRECTION: direction,
        }
    )


def test_weather_features_use_the_forecast_for_tomorrow():
    # Today 5 °C without sun; tomorrow 12 °C, 30 minutes of sun in every hour and
    # 10 km/h wind with one stronger hour at 15:00.
    today = forecast_hours("2026-10-10", temperature=5.0, sunshine=0.0)
    tomorrow = forecast_hours("2026-10-11", temperature=12.0, sunshine=1800.0)
    tomorrow.loc[15, features.WIND_SPEED] = 40.0

    row = features.weather_features(pd.concat([today, tomorrow], ignore_index=True)).loc[pd.Timestamp("2026-10-10")]

    assert row["air_temp_fc_d1"] == 12.0  # tomorrow's mean, not today's 5 °C
    assert row["sunshine_fc_d1"] == 12.0  # 24 hours x 1800 s = 12 hours of sun
    assert row["wind_max_fc_d1"] == 40.0  # the strongest hour of tomorrow


def test_south_west_component_of_the_wind():
    # 20 km/h from the south-west (225°), the north-east (45°, the bise) and the
    # south-east (135°, across the lake) on three days after 10.10.
    hours = pd.concat(
        [
            forecast_hours("2026-10-10"),
            forecast_hours("2026-10-11", speed=20.0, direction=225.0),
            forecast_hours("2026-10-12", speed=20.0, direction=45.0),
            forecast_hours("2026-10-13", speed=20.0, direction=135.0),
        ],
        ignore_index=True,
    )

    table = features.weather_features(hours)

    # The forecast for tomorrow, seen from the day before:
    assert table.loc[pd.Timestamp("2026-10-10"), "wind_sw_fc_d1"] == pytest.approx(20.0)  # from SW: full push
    assert table.loc[pd.Timestamp("2026-10-11"), "wind_sw_fc_d1"] == pytest.approx(-20.0)  # from NE: negative
    assert table.loc[pd.Timestamp("2026-10-12"), "wind_sw_fc_d1"] == pytest.approx(0.0, abs=1e-9)  # across: none
    # Today's value is the same number in the row of the day itself.
    assert table.loc[pd.Timestamp("2026-10-11"), "wind_sw_fc_d0"] == pytest.approx(20.0)


def test_wind_directions_are_not_averaged_as_angles():
    # Tomorrow the wind comes from 350° for 12 hours and from 10° for 12 hours:
    # both are almost north. The average of the angles would be 180° (south), which
    # would turn into a south-west push of about +7 km/h - the wrong sign.
    tomorrow = forecast_hours("2026-10-11", speed=10.0, direction=350.0)
    tomorrow.loc[12:, features.WIND_DIRECTION] = 10.0

    table = features.weather_features(pd.concat([forecast_hours("2026-10-10"), tomorrow], ignore_index=True))

    # Right: the mean of the hourly south-west parts, about -7 km/h (a north wind is
    # partly a north-east wind, so it counts negative).
    expected = 10 * (np.cos(np.deg2rad(350 - 225)) + np.cos(np.deg2rad(10 - 225))) / 2
    assert table.loc[pd.Timestamp("2026-10-10"), "wind_sw_fc_d1"] == pytest.approx(expected)
    assert expected < 0


def test_forecast_day_with_missing_hours_gives_no_weather_features():
    # Tomorrow has sunshine for only 19 of its 24 hours (one less than the rule's 20).
    tomorrow = forecast_hours("2026-10-11")
    tomorrow.loc[19:, features.SUNSHINE] = float("nan")

    row = features.weather_features(pd.concat([forecast_hours("2026-10-10"), tomorrow], ignore_index=True)).loc[
        pd.Timestamp("2026-10-10")
    ]

    # Every forecast for tomorrow is empty ...
    assert row[["air_temp_fc_d1", "sunshine_fc_d1", "wind_max_fc_d1", "wind_sw_fc_d1"]].isna().all()
    # ... but today's wind (a complete day) still exists.
    assert row["wind_sw_fc_d0"] == pytest.approx(10.0)


def test_load_weather_keeps_the_newest_download_and_ignores_the_snapshots(tmp_path):
    header = (
        "fetched_at_utc,time_utc,temperature_2m_previous_day2,sunshine_duration_previous_day2,"
        "wind_speed_10m_previous_day2,wind_direction_10m_previous_day2\n"
    )
    # The same hour (02.10 10:00 UTC) in two downloads, one day apart.
    (tmp_path / "weather_geneva_prev2_2026-10-01_to_2026-10-02.csv").write_text(
        header + "2026-10-01T19:00:00+00:00,2026-10-02T10:00,15.0,3600,10,225\n"
    )
    (tmp_path / "weather_geneva_prev2_2026-09-25_to_2026-10-03.csv").write_text(
        header + "2026-10-02T19:00:00+00:00,2026-10-02T10:00,15.5,3600,10,225\n"
    )
    # A snapshot of the newest forecast (downloaded after 18:00): it must not be read.
    (tmp_path / "weather_geneva_fetched_2026-10-02.csv").write_text(
        "fetched_at_utc,time_utc,temperature_2m\n2026-10-02T19:00:00+00:00,2026-10-02T10:00,99.0\n"
    )

    hours = features.load_weather(tmp_path)

    # One hour, from the newer download (02.10), although its file comes first in
    # name order; the 99 °C of the snapshot is not there.
    assert list(hours[features.TEMPERATURE]) == [15.5]
    # 10:00 UTC is 12:00 on the Swiss clock (summer time until 25.10.2026).
    assert hours["time_local"].iloc[0] == pd.Timestamp("2026-10-02 12:00", tz="Europe/Zurich")


def test_load_weather_undoes_the_open_meteo_shift_of_old_files(tmp_path):
    # An OLD file (time_local), downloaded on 27.10.2026 in winter time: Open-Meteo
    # shifted every hour by +1 h, also this summer hour, so "12:00" means 11:00 UTC.
    (tmp_path / "weather_geneva_prev2_2025-07-01_to_2025-07-01.csv").write_text(
        "fetched_at_utc,time_local,temperature_2m_previous_day2,sunshine_duration_previous_day2,"
        "wind_speed_10m_previous_day2,wind_direction_10m_previous_day2\n"
        "2026-10-27T11:13:00+00:00,2025-07-01T12:00,24.0,3600,10,225\n"
    )
    # A NEW file (time_utc) with the same hour, downloaded later.
    (tmp_path / "weather_geneva_prev2_2025-06-30_to_2025-07-02.csv").write_text(
        "fetched_at_utc,time_utc,temperature_2m_previous_day2,sunshine_duration_previous_day2,"
        "wind_speed_10m_previous_day2,wind_direction_10m_previous_day2\n"
        "2026-10-28T19:00:00+00:00,2025-07-01T11:00,24.0,3600,10,225\n"
    )

    hours = features.load_weather(tmp_path)

    # Both describe 11:00 UTC, so they are one hour after loading ...
    assert len(hours) == 1
    assert hours["time_utc"].iloc[0] == pd.Timestamp("2025-07-01 11:00", tz="UTC")
    # ... which is 13:00 on the Swiss clock in summer (UTC+2), not 12:00.
    assert hours["time_local"].iloc[0] == pd.Timestamp("2025-07-01 13:00", tz="Europe/Zurich")


def test_feature_table_has_all_features_and_the_label(tmp_path):
    # One raw folder with the real water sample and a fake forecast for the same days.
    (tmp_path / "water_2606_sample.csv").write_text((FIXTURES / "water_2606_sample.csv").read_text())
    hours = pd.concat([forecast_hours(f"2026-03-{d}") for d in range(27, 32)], ignore_index=True)
    # Saved like a real file: the UTC time only (time_local is computed when reading).
    hours.drop(columns="time_local").to_csv(tmp_path / "weather_geneva_prev2_2026-03-27_to_2026-03-31.csv", index=False)

    table = features.feature_table(tmp_path)

    # The columns in their fixed order: counts, all features, label.
    assert list(table.columns) == ["n_readings_d0", *features.FEATURES, "n_readings_d1", features.LABEL]
    # One row per water day; 30.03 has its forecast for 31.03, so every feature exists.
    assert len(table) == 4
    assert table.loc[pd.Timestamp("2026-03-30"), features.FEATURES].notna().all()
