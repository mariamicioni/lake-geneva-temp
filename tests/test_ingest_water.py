"""Tests for the water download script. They do not use the internet."""
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

from lakegeneva.features import ingest_water


def test_fetch_combines_temperature_and_flow_per_timestamp(monkeypatch):
    # fake API answer: two readings at the same time, one at a later time
    # (1760000600 is 600 seconds = 10 minutes after 1760000000)
    payload = {"payload": [
        {"timestamp": 1760000000, "par": "temperature", "val": 18.5},
        {"timestamp": 1760000000, "par": "flow", "val": 250.0},
        {"timestamp": 1760000600, "par": "temperature", "val": 18.4},
    ]}

    # A stand-in for urllib.request.urlopen: instead of calling the API it returns
    # the fake answer as bytes, like a real HTTP response.
    def fake_urlopen(request, timeout):
        return io.BytesIO(json.dumps(payload).encode())

    # monkeypatch (a pytest helper) swaps urlopen for the fake during this test only.
    monkeypatch.setattr(ingest_water.urllib.request, "urlopen", fake_urlopen)

    readings = ingest_water.fetch(dt.date(2025, 10, 9), dt.date(2025, 10, 10))

    # Both values of the same time end up in one entry; the later time has no flow.
    assert readings == {
        1760000000: {"temperature": 18.5, "flow": 250.0},
        1760000600: {"temperature": 18.4},
    }


def test_write_csv_sorts_by_time_and_leaves_missing_values_empty(tmp_path):
    # Readings given in the wrong order, and the later one has no flow.
    readings = {
        1760000600: {"temperature": 18.4},
        1760000000: {"temperature": 18.5, "flow": 250.0},
    }
    # tmp_path (a pytest helper) is a fresh temporary folder, deleted afterwards,
    # so the test never writes into the real data/ folder.
    path = tmp_path / "water.csv"

    ingest_water.write_csv(readings, path)

    # Read the file back and compare it line by line.
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [
        ["time_utc", "water_temp_c", "flow_m3s"],
        ["2025-10-09T08:53:20+00:00", "18.5", "250.0"],  # Unix time -> readable UTC, oldest first
        ["2025-10-09T09:03:20+00:00", "18.4", ""],  # missing flow -> empty cell, not 0
    ]


def test_fetch_bafu_gives_the_same_shape_as_existenz(monkeypatch):
    # fake BAFU answer: WT and Q at the same time, rows in any order
    answer = {"data": {"water": {"observations": {"data_10min_mean": [
        {"timestamp": "2025-10-09T08:53:20Z", "parameterName": "Q", "value": 250.0},
        {"timestamp": "2025-10-09T08:53:20Z", "parameterName": "WT", "value": 18.5},
        {"timestamp": "2025-10-09T09:03:20Z", "parameterName": "WT", "value": 18.4},
    ]}}}}

    def fake_urlopen(request, timeout):
        return io.BytesIO(json.dumps(answer).encode())

    monkeypatch.setattr(ingest_water.urllib.request, "urlopen", fake_urlopen)

    readings = ingest_water.fetch_bafu(dt.date(2025, 10, 9), dt.date(2025, 10, 10))

    # Same result as the existenz.ch test above: Unix time keys, existenz.ch names.
    assert readings == {
        1760000000: {"temperature": 18.5, "flow": 250.0},
        1760000600: {"temperature": 18.4},
    }


def test_main_falls_back_to_bafu_when_existenz_fails(monkeypatch, tmp_path):
    # existenz.ch is "down": fetch() raises the error seen on GitHub on 21.10.2026.
    def broken_fetch(start, end):
        raise OSError("[Errno 101] Network is unreachable")

    monkeypatch.setattr(ingest_water, "fetch", broken_fetch)
    monkeypatch.setattr(ingest_water, "fetch_bafu", lambda start, end: {1760000000: {"temperature": 18.5}})
    # Pretend the script was started with these command-line options.
    monkeypatch.setattr("sys.argv", ["ingest_water.py", "--out", str(tmp_path)])

    ingest_water.main()

    # The CSV was still written, with the reading from the fallback.
    [path] = tmp_path.glob("water_2606_fetched_*.csv")
    assert "2025-10-09T08:53:20+00:00,18.5," in path.read_text()
