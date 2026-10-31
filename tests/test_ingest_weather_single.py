"""Tests for the download of the 06:00 UTC model run. They do not use the internet."""
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
import urllib.error

import pytest  # pytest.raises: checks that an error really happens

from lakegeneva.features import ingest_weather_single


def test_fetch_run_asks_for_the_06_utc_run_of_the_day(monkeypatch):
    # A stand-in for urllib.request.urlopen: it remembers the URL and returns a tiny answer.
    urls = []

    def fake_urlopen(request, timeout):
        urls.append(request.full_url)
        return io.BytesIO(json.dumps({"hourly": {"time": []}}).encode())

    # monkeypatch (a pytest helper) swaps urlopen for the fake during this test only.
    monkeypatch.setattr(ingest_weather_single.urllib.request, "urlopen", fake_urlopen)

    ingest_weather_single.fetch_run(dt.date(2026, 10, 27))

    [url] = urls  # exactly one request was sent
    assert "run=2026-10-27T06:00" in url  # the 06:00 UTC run of that day
    assert "models=ecmwf_ifs" in url  # ECMWF IFS, the model with the longest archive
    assert "forecast_hours=48" in url  # long enough for every hour of tomorrow
    assert "timezone=GMT" in url  # forecast hours in UTC, like all raw data


def test_a_run_missing_in_the_archive_is_skipped(monkeypatch):
    # The real answer for a missing run (checked on 27.10.2026): HTTP 400 with this reason.
    def fake_urlopen(request, timeout):
        body = io.BytesIO(b'{"error":true,"reason":"The requested model run is not available."}')
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, body)

    monkeypatch.setattr(ingest_weather_single.urllib.request, "urlopen", fake_urlopen)

    # None means "skip this day", not an error.
    assert ingest_weather_single.fetch_run(dt.date(2025, 8, 5)) is None


def test_other_http_errors_are_tried_three_times_and_then_raised(monkeypatch):
    # The API is "overloaded" (HTTP 503) on every attempt.
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 503, "Service Unavailable", {}, io.BytesIO(b"busy"))

    monkeypatch.setattr(ingest_weather_single.urllib.request, "urlopen", fake_urlopen)
    # No 60-second waits during the test.
    monkeypatch.setattr(ingest_weather_single.time, "sleep", lambda seconds: None)

    # pytest.raises: the test passes only if this error really happens.
    with pytest.raises(urllib.error.HTTPError):
        ingest_weather_single.fetch_run(dt.date(2026, 10, 27))
    assert len(calls) == 3  # three attempts, then the job fails visibly


def test_main_writes_one_file_and_skips_missing_days(monkeypatch, tmp_path, capsys):
    # Three issue days; the run of the middle one is missing in the archive.
    def fake_fetch_run(day):
        if day == dt.date(2025, 8, 5):
            return None
        return {"time": ["2025-08-05T00:00", "2025-08-05T01:00"], **{v: [1.0, None] for v in ingest_weather_single.VARIABLES}}

    monkeypatch.setattr(ingest_weather_single, "fetch_run", fake_fetch_run)
    monkeypatch.setattr(ingest_weather_single.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        "sys.argv", ["ingest_weather_single.py", "--start", "2025-08-04", "--end", "2025-08-06", "--out", str(tmp_path)]
    )

    ingest_weather_single.main()

    # One file for the range: a header plus 2 runs x 2 hours.
    with open(tmp_path / "weather_geneva_single06_2025-08-04_to_2025-08-06.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1 + 4
    # Every row says which run it comes from; the missing day is not there.
    assert {row[1] for row in rows[1:]} == {"2025-08-04T06:00Z", "2025-08-06T06:00Z"}
    # A missing value becomes an empty cell, not 0.
    assert rows[1][3:] == ["1.0"] * 4 and rows[2][3:] == [""] * 4
    # capsys (a pytest helper) catches what the script printed: the skipped day is named.
    assert "2025-08-05" in capsys.readouterr().out


def test_a_second_run_with_the_same_range_keeps_the_first_file(monkeypatch, tmp_path):
    answer = {"time": ["2025-08-05T00:00"], **{v: [1.0] for v in ingest_weather_single.VARIABLES}}
    monkeypatch.setattr(ingest_weather_single.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        "sys.argv", ["ingest_weather_single.py", "--start", "2025-08-04", "--end", "2025-08-05", "--out", str(tmp_path)]
    )

    # The first run gets both days, the second run only gets 05.08.
    monkeypatch.setattr(ingest_weather_single, "fetch_run", lambda day: answer)
    ingest_weather_single.main()
    monkeypatch.setattr(ingest_weather_single, "fetch_run", lambda day: answer if day == dt.date(2025, 8, 5) else None)
    ingest_weather_single.main()

    # The first file still holds the run of 04.08; the second run has its own file.
    first = tmp_path / "weather_geneva_single06_2025-08-04_to_2025-08-05.csv"
    assert "2025-08-04T06:00Z" in first.read_text()
    assert (tmp_path / "weather_geneva_single06_2025-08-04_to_2025-08-05_run02.csv").exists()


def test_main_daily_run_asks_for_today_and_the_6_days_before(monkeypatch, tmp_path):
    # A stand-in for fetch_run() that notes which days were asked for.
    asked = []

    def fake_fetch_run(day):
        asked.append(day)
        return {"time": [], **{v: [] for v in ingest_weather_single.VARIABLES}}

    monkeypatch.setattr(ingest_weather_single, "fetch_run", fake_fetch_run)
    monkeypatch.setattr(ingest_weather_single.time, "sleep", lambda seconds: None)
    # Without --start and --end the script does the daily run.
    monkeypatch.setattr("sys.argv", ["ingest_weather_single.py", "--out", str(tmp_path)])

    ingest_weather_single.main()

    today = dt.datetime.now(dt.timezone.utc).date()
    assert asked == [today - dt.timedelta(days=n) for n in range(6, -1, -1)]  # 7 days, oldest first
