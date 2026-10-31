"""Tests for the download of 48-hour-old forecasts. They do not use the internet."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push
# (.github/workflows/ci.yml). pytest finds every function whose name starts with
# test_ and runs it; a test fails if one of its `assert` lines is false.
#
# Why no internet: tests must give the same result every time. A real API call
# can fail or return new data, so we replace the API with a fixed fake answer.
import csv
import datetime as dt
import io
import json

import pytest

from lakegeneva.common import features
from lakegeneva.features import ingest_weather_runs


def test_fetch_asks_for_the_48_hour_old_values(monkeypatch):
    # A stand-in for urllib.request.urlopen: it remembers the URL it was called
    # with and returns a tiny answer as bytes, like a real HTTP response.
    urls = []

    def fake_urlopen(request, timeout):
        urls.append(request.full_url)
        return io.BytesIO(json.dumps({"hourly": {"time": []}}).encode())

    # monkeypatch (a pytest helper) swaps urlopen for the fake during this test only.
    monkeypatch.setattr(ingest_weather_runs.urllib.request, "urlopen", fake_urlopen)

    ingest_weather_runs.fetch(dt.date(2024, 1, 20), dt.date(2024, 1, 21))

    [url] = urls  # exactly one request was sent
    # Every variable is asked for as its 48-hour-old version ...
    for variable in ["temperature_2m", "sunshine_duration", "wind_speed_10m", "wind_direction_10m"]:
        assert f"{variable}_previous_day2" in url
    # ... for the requested days, in UTC (features.py converts to Swiss time).
    assert "start_date=2024-01-20&end_date=2024-01-21" in url
    assert "timezone=GMT" in url


def test_write_csv_has_one_row_per_hour_and_leaves_missing_values_empty(tmp_path):
    # Two hours of a fake answer; sunshine is missing (None) in the first hour,
    # as it is before 20.01.2024.
    hourly = {
        "time": ["2024-01-20T12:00", "2024-01-20T13:00"],
        "temperature_2m_previous_day2": [4.1, 4.5],
        "sunshine_duration_previous_day2": [None, 1800.0],
        "wind_speed_10m_previous_day2": [10.2, 12.0],
        "wind_direction_10m_previous_day2": [225, 230],
    }
    fetched_at = dt.datetime(2026, 10, 27, 19, 30, tzinfo=dt.timezone.utc)
    # tmp_path (a pytest helper) is a fresh temporary folder, deleted afterwards,
    # so the test never writes into the real data/ folder.
    path = tmp_path / "weather.csv"

    ingest_weather_runs.write_csv(hourly, fetched_at, path)

    # Read the file back and compare it line by line.
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [
        [
            "fetched_at_utc",
            "time_utc",
            "temperature_2m_previous_day2",
            "sunshine_duration_previous_day2",
            "wind_speed_10m_previous_day2",
            "wind_direction_10m_previous_day2",
        ],
        ["2026-10-27T19:30:00+00:00", "2024-01-20T12:00", "4.1", "", "10.2", "225"],  # missing -> empty cell, not 0
        ["2026-10-27T19:30:00+00:00", "2024-01-20T13:00", "4.5", "1800.0", "12.0", "230"],
    ]


def test_main_daily_run_includes_tomorrow(monkeypatch, tmp_path):
    # A stand-in for fetch() that notes the range it was asked for.
    asked = []

    def fake_fetch(start, end):
        asked.append((start, end))
        # One hour is enough: this test only checks the range.
        return {"time": ["2026-10-30T00:00"], **{column: [1.0] for column in ingest_weather_runs.COLUMNS}}

    monkeypatch.setattr(ingest_weather_runs, "fetch", fake_fetch)
    # Without --start and --end the script does the daily run.
    monkeypatch.setattr("sys.argv", ["ingest_weather_runs.py", "--out", str(tmp_path)])

    ingest_weather_runs.main()

    # Tomorrow (the day we predict) is included, plus the 7 days before today.
    today = dt.datetime.now(dt.timezone.utc).date()
    assert asked == [(today - dt.timedelta(days=7), today + dt.timedelta(days=1))]


def test_main_backfill_writes_one_file_for_the_range(monkeypatch, tmp_path):
    # A stand-in for fetch() that notes the range and returns one hour.
    asked = []

    def fake_fetch(start, end):
        asked.append((start, end))
        return {"time": ["2024-01-21T00:00"], **{column: [1.0] for column in ingest_weather_runs.COLUMNS}}

    monkeypatch.setattr(ingest_weather_runs, "fetch", fake_fetch)
    # Pretend the script was started for a backfill.
    monkeypatch.setattr(
        "sys.argv", ["ingest_weather_runs.py", "--start", "2024-01-20", "--end", "2026-10-28", "--out", str(tmp_path)]
    )

    ingest_weather_runs.main()

    # The days are passed on unchanged (as UTC days).
    assert asked == [(dt.date(2024, 1, 20), dt.date(2026, 10, 28))]
    # One file named after the range.
    assert (tmp_path / "weather_geneva_prev2_2024-01-20_to_2026-10-28.csv").exists()


def test_a_second_download_of_the_same_range_loses_no_value(monkeypatch, tmp_path):
    # Two downloads of the same range on one day. The first answer is complete, the
    # second one has no sunshine (not all empty, so the script accepts it).
    hours = ["2026-11-01T10:00", "2026-11-01T11:00"]
    complete = {"time": hours, **{column: [10.0, 10.0] for column in ingest_weather_runs.COLUMNS}}
    gaps = {**complete, features.SUNSHINE: [None, None]}
    answers = iter([complete, gaps])
    monkeypatch.setattr(ingest_weather_runs, "fetch", lambda start, end: next(answers))
    monkeypatch.setattr(
        "sys.argv", ["ingest_weather_runs.py", "--start", "2026-10-25", "--end", "2026-11-01", "--out", str(tmp_path)]
    )

    ingest_weather_runs.main()
    ingest_weather_runs.main()

    # The second download got its own file, so the first file is still complete ...
    assert (tmp_path / "weather_geneva_prev2_2026-10-25_to_2026-11-01.csv").exists()
    assert (tmp_path / "weather_geneva_prev2_2026-10-25_to_2026-11-01_run02.csv").exists()
    # ... and the shared loader still finds every sunshine value.
    assert list(features.load_weather(tmp_path)[features.SUNSHINE]) == [10.0, 10.0]


@pytest.mark.parametrize(
    "answer",
    [
        {"time": []},  # no hours at all
        # Hours, but every value is empty.
        {"time": ["2026-10-30T00:00"], **{column: [None] for column in ingest_weather_runs.COLUMNS}},
    ],
)
def test_main_stops_and_writes_no_file_when_there_are_no_values(monkeypatch, tmp_path, answer):
    monkeypatch.setattr(ingest_weather_runs, "fetch", lambda start, end: answer)
    monkeypatch.setattr("sys.argv", ["ingest_weather_runs.py", "--out", str(tmp_path)])

    # pytest.raises: the test passes only if the script stops with an error.
    with pytest.raises(SystemExit):
        ingest_weather_runs.main()

    # No file: an empty file would hide the problem, and it would break a build that
    # has no older files (the Docker image).
    assert list(tmp_path.glob("*.csv")) == []
