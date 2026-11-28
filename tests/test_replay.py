"""Tests for the replay mode (common/web.py): the whole pipeline on saved API answers, no internet."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push. CI also runs
# the Docker image on the same saved answers (.github/workflows/ci.yml, MS2 review A3).
import runpy  # runs the check script of the fixtures folder and returns its functions
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

from lakegeneva.common import features, web
from lakegeneva.features import run_pipeline

ANSWERS = Path(__file__).parent / "fixtures" / "replay"


def no_internet(*args, **kwargs):
    raise AssertionError("the replay mode must not use the internet")


def test_each_api_has_one_saved_answer_by_host_and_path():
    # The query (dates, point) is left out: it changes every day.
    assert web.answer_name("https://data.bafu.admin.ch/api") == "data.bafu.admin.ch_api.json"
    url = "https://previous-runs-api.open-meteo.com/v1/forecast?latitude=46.2&start_date=2026-11-20"
    assert web.answer_name(url) == "previous-runs-api.open-meteo.com_v1_forecast.json"
    # Every API of the daily run has its answer in the fixtures folder.
    for name in (
        "data.bafu.admin.ch_api.json",
        "previous-runs-api.open-meteo.com_v1_forecast.json",
        "api.open-meteo.com_v1_forecast.json",
        "single-runs-api.open-meteo.com_v1_forecast.json",
    ):
        assert (ANSWERS / name).is_file(), name


def test_without_the_variable_the_request_goes_to_the_web(monkeypatch):
    monkeypatch.delenv(web.REPLAY, raising=False)
    sent = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: sent.append(request.full_url) or "answer")

    assert web.open_url(urllib.request.Request("https://api.open-meteo.com/v1/forecast?x=1"), timeout=5) == "answer"
    assert sent == ["https://api.open-meteo.com/v1/forecast?x=1"]


def test_a_missing_answer_stops_at_once(monkeypatch, tmp_path):
    # Not an OSError: the download scripts would retry that three times with a minute
    # of waiting, and a missing file does not appear by waiting.
    monkeypatch.setenv(web.REPLAY, str(tmp_path))
    monkeypatch.setattr(urllib.request, "urlopen", no_internet)

    with pytest.raises(web.MissingAnswer, match="data.bafu.admin.ch_api.json") as stop:
        web.open_url(urllib.request.Request("https://data.bafu.admin.ch/api"), timeout=5)
    assert not isinstance(stop.value, OSError)


def test_the_whole_pipeline_runs_on_the_saved_answers_without_internet(monkeypatch, tmp_path):
    # The same as CI's image check: all five steps of run_pipeline, in an empty folder,
    # with the internet switched off. A change that breaks a download or the table fails here.
    monkeypatch.setenv(web.REPLAY, str(ANSWERS))
    monkeypatch.setattr(urllib.request, "urlopen", no_internet)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_pipeline"])

    run_pipeline.main()

    # One raw file per download, and the feature table built from them.
    assert len(list((tmp_path / "data" / "raw").glob("*.csv"))) == 4
    table = pd.read_parquet(tmp_path / "data" / "features" / "features.parquet")
    # The same check as in CI's image job (tests/fixtures/replay/check_table.py): the
    # issue day 26.11, all ten features on its row, and some rows with a label.
    assert check_table(table).startswith("table OK: issue day 2026-11-26 with all 10 features")
    # And the check really fails when a feature is missing on the issue day (review of
    # 27.11.2026: without wind the date was still right, and the old check stayed green).
    broken = table.copy()
    broken.loc[features.issue_day(table), "wind_sw_fc_d0"] = None
    with pytest.raises(SystemExit, match="misses wind_sw_fc_d0"):
        check_table(broken)
    unlabelled = table.assign(**{features.LABEL: None})
    with pytest.raises(SystemExit, match="no row has a label"):
        check_table(unlabelled)
    # Labels, but every labelled row misses a feature: no row to learn from (review of
    # 27.11.2026: this passed before). The row of the issue day has no label, so it stays complete.
    no_training_row = table.copy()
    no_training_row.loc[table[features.LABEL].notna(), "wind_max_fc_d1"] = None
    with pytest.raises(SystemExit, match="none has all 10 features too"):
        check_table(no_training_row)


def check_table(table: pd.DataFrame) -> str:
    """The check function of tests/fixtures/replay/check_table.py (a plain script, not a module)."""
    return runpy.run_path(str(ANSWERS / "check_table.py"))["check"](table)
