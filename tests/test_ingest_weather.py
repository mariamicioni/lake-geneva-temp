"""Tests for the audit copy of the newest forecast. They do not use the internet."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push
# (.github/workflows/ci.yml). pytest finds every function whose name starts with
# test_ and runs it; a test fails if one of its `assert` lines is false.
#
# Why no internet: tests must give the same result every time. A real API call
# can fail or return new data, so we replace the API with a fixed fake answer.
from lakegeneva.features import ingest_weather


def test_a_second_run_on_the_same_day_keeps_the_first_snapshot(monkeypatch, tmp_path):
    # The newest forecast changes during the day, so the two answers differ.
    def answer(value):
        return {"time": ["2026-10-31T12:00"], **{variable: [value] for variable in ingest_weather.VARIABLES}}

    answers = iter([answer(1.0), answer(2.0)])
    # monkeypatch (a pytest helper) swaps fetch() for the fake during this test only.
    monkeypatch.setattr(ingest_weather, "fetch", lambda: next(answers))
    monkeypatch.setattr("sys.argv", ["ingest_weather.py", "--out", str(tmp_path)])

    ingest_weather.main()
    ingest_weather.main()

    # Two files, in the order of the runs: this forecast can never be downloaded again,
    # so the first copy must still be there.
    first, second = sorted(tmp_path.glob("weather_geneva_fetched_*.csv"))
    assert ",1.0," in first.read_text()
    assert ",2.0," in second.read_text()
