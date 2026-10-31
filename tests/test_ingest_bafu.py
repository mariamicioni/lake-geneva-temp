"""Tests for the BAFU daily-mean download. They do not use the internet."""
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

from lakegeneva.features import ingest_bafu


def test_query_covers_the_whole_first_and_last_day():
    # Ask for 01.10.2026 to 10.10.2026 and check the time window in the query text.
    query = ingest_bafu.build_query(dt.date(2026, 10, 1), dt.date(2026, 10, 10))

    # midnight Swiss winter time = 23:00 UTC of the day before
    assert '_gte: "2026-09-30T23:00:00Z"' in query  # starts with 01.10 included
    assert '_lt: "2026-10-10T23:00:00Z"' in query  # stops before 11.10, so 10.10 is included
    assert '"WT", "Q"' in query  # asks for water temperature and discharge


def test_fetch_turns_bafu_timestamps_into_swiss_days(monkeypatch):
    # fake API answer, rows in any order as the real API sends them
    answer = {"data": {"water": {"observations": {"data_1day_mean": [
        {"timestamp": "2026-10-09T23:00:00Z", "parameterName": "WT", "value": 19.04},
        {"timestamp": "2026-10-08T23:00:00Z", "parameterName": "Q", "value": 184.609},
        {"timestamp": "2026-10-09T23:00:00Z", "parameterName": "Q", "value": 124.946},
        {"timestamp": "2026-10-08T23:00:00Z", "parameterName": "WT", "value": 19.27},
    ]}}}}

    # A stand-in for urllib.request.urlopen: instead of calling the API it returns
    # the fake answer as bytes, like a real HTTP response.
    def fake_urlopen(request, timeout):
        return io.BytesIO(json.dumps(answer).encode())

    # monkeypatch (a pytest helper) swaps urlopen for the fake during this test only.
    monkeypatch.setattr(ingest_bafu.urllib.request, "urlopen", fake_urlopen)

    days = ingest_bafu.fetch(dt.date(2026, 10, 9), dt.date(2026, 10, 10))

    # 23:00Z on the 8th is the mean of the 9th, 23:00Z on the 9th the mean of the
    # 10th, and both parameters of a day end up in one entry.
    assert days == {
        dt.date(2026, 10, 9): {"water_temp_c": 19.27, "flow_m3s": 184.609},
        dt.date(2026, 10, 10): {"water_temp_c": 19.04, "flow_m3s": 124.946},
    }


def test_write_csv_sorts_by_day_and_leaves_missing_values_empty(tmp_path):
    # Days given in the wrong order, and the flow of 10.10 is missing.
    days = {
        dt.date(2026, 10, 10): {"water_temp_c": 19.04},
        dt.date(2026, 10, 9): {"water_temp_c": 19.27, "flow_m3s": 184.609},
    }
    # tmp_path (a pytest helper) is a fresh temporary folder, deleted afterwards,
    # so the test never writes into the real data/ folder.
    path = tmp_path / "bafu.csv"

    ingest_bafu.write_csv(days, path)

    # Read the file back and compare it line by line.
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [
        ["date", "water_temp_c", "flow_m3s"],
        ["2026-10-09", "19.27", "184.609"],  # sorted: oldest day first
        ["2026-10-10", "19.04", ""],  # missing flow -> empty cell, not 0
    ]


def test_a_second_run_with_the_same_range_keeps_the_first_file(monkeypatch, tmp_path):
    # Two runs for the same day. The second answer has no flow.
    day = dt.date(2026, 10, 10)
    answers = iter([{day: {"water_temp_c": 19.04, "flow_m3s": 124.946}}, {day: {"water_temp_c": 19.04}}])
    monkeypatch.setattr(ingest_bafu, "fetch", lambda start, end: next(answers))
    monkeypatch.setattr(
        "sys.argv", ["ingest_bafu.py", "--start", "2026-10-10", "--end", "2026-10-10", "--out", str(tmp_path)]
    )

    ingest_bafu.main()
    ingest_bafu.main()

    # The first file still has the flow; the second run has its own file.
    assert "124.946" in (tmp_path / "bafu_2606_daily_2026-10-10_to_2026-10-10.csv").read_text()
    assert (tmp_path / "bafu_2606_daily_2026-10-10_to_2026-10-10_run02.csv").exists()
