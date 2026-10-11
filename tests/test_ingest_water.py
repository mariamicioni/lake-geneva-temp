"""Tests for the water download script. They do not use the internet."""
import csv
import datetime as dt
import io
import json

from lakegeneva.features import ingest_water


def test_fetch_combines_temperature_and_flow_per_timestamp(monkeypatch):
    # fake API answer: two readings at the same time, one at a later time
    payload = {"payload": [
        {"timestamp": 1760000000, "par": "temperature", "val": 18.5},
        {"timestamp": 1760000000, "par": "flow", "val": 250.0},
        {"timestamp": 1760000600, "par": "temperature", "val": 18.4},
    ]}

    def fake_urlopen(request, timeout):
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(ingest_water.urllib.request, "urlopen", fake_urlopen)

    readings = ingest_water.fetch(dt.date(2025, 10, 9), dt.date(2025, 10, 10))

    assert readings == {
        1760000000: {"temperature": 18.5, "flow": 250.0},
        1760000600: {"temperature": 18.4},
    }


def test_write_csv_sorts_by_time_and_leaves_missing_values_empty(tmp_path):
    readings = {
        1760000600: {"temperature": 18.4},
        1760000000: {"temperature": 18.5, "flow": 250.0},
    }
    path = tmp_path / "water.csv"

    ingest_water.write_csv(readings, path)

    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [
        ["time_utc", "water_temp_c", "flow_m3s"],
        ["2025-10-09T08:53:20+00:00", "18.5", "250.0"],
        ["2025-10-09T09:03:20+00:00", "18.4", ""],
    ]
