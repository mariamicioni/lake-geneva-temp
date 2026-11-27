"""Tests for check_fresh.py: the row of the issue day D for the service evening S (MS2 review point A2)."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push.
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

from lakegeneva.common import features
from lakegeneva.features import check_fresh

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "feature-pipeline.yml"
EVENING = dt.date(2026, 11, 25)  # the service evening S of most tests


def table_until(last_day: str, readings=(108, 108, 108)) -> pd.DataFrame:
    """Three days up to last_day, all features filled, `readings` before 18:00 on each day."""
    days = pd.date_range(end=last_day, periods=3, freq="D", name="date")
    table = pd.DataFrame(1.0, index=days, columns=features.FEATURES)
    table["n_readings_d0"] = list(readings)
    return table


def utc(text: str) -> dt.datetime:
    """A time in UTC, e.g. utc("2026-11-26 02:05")."""
    return dt.datetime.fromisoformat(text).replace(tzinfo=dt.UTC)


def test_the_service_evening_is_the_newest_scheduled_start_that_has_passed():
    # The run of 25.11 committed after midnight (03:05 Swiss = 02:05 UTC): still 25.11.
    assert check_fresh.service_day(utc("2026-11-26 02:05")) == EVENING
    # 19:30 UTC: the job of 25.11 is due ...
    assert check_fresh.service_day(utc("2026-11-25 19:30")) == EVENING
    # ... at 18:30 UTC not yet, so it is the evening before.
    assert check_fresh.service_day(utc("2026-11-25 18:30")) == dt.date(2026, 11, 24)
    # A Monday retrain that GitHub starts late, in summer (16:44 UTC = 18:44 Swiss):
    # Sunday, whose job has run, not Monday, whose job is not due yet (check of 26.11.2026).
    assert check_fresh.service_day(utc("2026-06-29 16:44")) == dt.date(2026, 6, 28)
    # The same in winter, as on 23.11.2026.
    assert check_fresh.service_day(utc("2026-11-23 16:44")) == dt.date(2026, 11, 22)


def test_the_scheduled_start_is_the_one_of_the_daily_job():
    # If someone moves the cron of the daily job, the service evening must move with it.
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert f'cron: "0 {check_fresh.SCHEDULE_HOUR_UTC} * * *"' in workflow


def test_on_time_when_the_issue_day_is_the_service_evening():
    passed, delayed, text = check_fresh.check_issue_day(table_until("2026-11-25"), EVENING)

    assert passed and not delayed
    assert text.startswith("on time") and "D 25.11.2026" in text


def test_one_day_late_passes_but_says_delayed():
    # 25.11 has only 78 readings before 18:00 at run time (as 19.11.2026 had): D is 24.11.
    table = table_until("2026-11-25", readings=(108, 108, 78))

    passed, delayed, text = check_fresh.check_issue_day(table, EVENING)

    assert passed and delayed
    assert "only 78 of the 90" in text
    # The row of a day forecasts the next day: the inputs for 26.11 are missing, and the
    # row of D (24.11) forecasts 25.11 (review of 27.11.2026: a first text got this wrong).
    assert "forecast of 26.11.2026 are missing" in text
    assert "issue day D 24.11.2026, which forecasts 25.11.2026" in text


def test_two_days_late_fails():
    # Neither 25.11 nor 24.11 has 90 readings: D is 23.11.
    passed, _, text = check_fresh.check_issue_day(table_until("2026-11-25", readings=(108, 60, 40)), EVENING)
    assert not passed and "2 days before" in text
    # The same when the table has no row of S at all yet (a run that stopped early).
    passed, _, text = check_fresh.check_issue_day(table_until("2026-11-23"), EVENING)
    assert not passed and "2 days before" in text


def test_a_feature_missing_on_the_issue_day_fails_although_an_older_row_is_complete():
    # The case the old age rule let through: the row of D misses tomorrow's wind, the
    # row before is complete (age 1 day, "fresh").
    table = table_until("2026-11-25")
    table.loc[pd.Timestamp("2026-11-25"), "wind_sw_fc_d1"] = None

    passed, _, text = check_fresh.check_issue_day(table, EVENING)

    assert not passed and "misses wind_sw_fc_d1" in text


def test_no_complete_day_means_no_issue_day():
    passed, _, text = check_fresh.check_issue_day(table_until("2026-11-25", readings=(10, 10, 10)), EVENING)

    assert not passed and "no issue day" in text


def test_other_broken_tables_fail():
    # An infinite value is not a real feature either.
    table = table_until("2026-11-25")
    table.loc[pd.Timestamp("2026-11-25"), "air_temp_fc_d1"] = float("inf")
    assert not check_fresh.check_issue_day(table, EVENING)[0]
    # A missing count of readings (NaN) is "not complete": D is the day before.
    table = table_until("2026-11-25", readings=(108, 108, None))
    passed, delayed, _ = check_fresh.check_issue_day(table, EVENING)
    assert passed and delayed
    # Two rows for one date: not clear which is the row of D.
    doubled = pd.concat([table_until("2026-11-25"), table_until("2026-11-25").tail(1)])
    passed, _, text = check_fresh.check_issue_day(doubled, EVENING)
    assert not passed and "exactly one row per date" in text
    # A row at another time of the day: not one row per whole date.
    shifted = table_until("2026-11-25")
    shifted.index = shifted.index + pd.Timedelta(hours=12)
    passed, _, text = check_fresh.check_issue_day(shifted, EVENING)
    assert not passed and "one row per date, at midnight" in text
    # An infinite count of readings would pass "at least 90" (review of 27.11.2026).
    passed, _, text = check_fresh.check_issue_day(table_until("2026-11-25", readings=(108, 108, float("inf"))), EVENING)
    assert not passed and "infinite" in text


def test_rows_after_the_service_evening_do_not_count():
    # A table one day newer than S (a check the next afternoon), and the row of S itself
    # misses the wind. A first version took the newest complete day (26.11) and said
    # "on time" (review of 27.11.2026). D for S is the newest complete day UP TO S.
    table = table_until("2026-11-26")
    table.loc[pd.Timestamp("2026-11-25"), "wind_sw_fc_d0"] = None

    passed, _, text = check_fresh.check_issue_day(table, EVENING)

    assert not passed and "D 25.11.2026 misses wind_sw_fc_d0" in text
    # With a complete row of S, a newer day changes nothing: on time, D is S.
    passed, delayed, text = check_fresh.check_issue_day(table_until("2026-11-26"), EVENING)
    assert passed and not delayed and "issue day D 25.11.2026" in text
    # A later row without readings yet (like a weather-only row of tomorrow) changes nothing either.
    passed, delayed, _ = check_fresh.check_issue_day(table_until("2026-11-26", readings=(108, 108, 0)), EVENING)
    assert passed and not delayed


def run_main(monkeypatch, tmp_path, table, *options) -> None:
    """Save the table as Parquet and start main() with these command-line options."""
    path = tmp_path / "features.parquet"
    table.to_parquet(path)
    monkeypatch.setattr(sys, "argv", ["check_fresh", "--table", str(path), *options])
    check_fresh.main()


def test_main_shows_a_warning_on_github_when_the_issue_day_is_late(monkeypatch, tmp_path, capsys):
    late = table_until("2026-11-25", readings=(108, 108, 78))

    # On GitHub: a warning line first, which GitHub shows in yellow in the run summary.
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    run_main(monkeypatch, tmp_path, late, "--service-day", "2026-11-25")
    assert capsys.readouterr().out.startswith("::warning title=Issue day delayed::delayed")
    # On a laptop: the same text, without the warning line. The exit code is 0 both times.
    monkeypatch.delenv("GITHUB_ACTIONS")
    run_main(monkeypatch, tmp_path, late, "--service-day", "2026-11-25")
    out = capsys.readouterr().out
    assert out.startswith("delayed") and "::warning" not in out


def test_require_on_time_lets_only_the_row_of_s_pass(monkeypatch, tmp_path, capsys):
    # The question of the catch-up run (MS2 review point E4): is the evening on time?
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    run_main(monkeypatch, tmp_path, table_until("2026-11-25"), "--service-day", "2026-11-25", "--require-on-time")
    assert capsys.readouterr().out.startswith("on time")
    # One day late: without the option green, with it red, with the reason.
    late = table_until("2026-11-25", readings=(108, 108, 78))
    with pytest.raises(SystemExit) as stop:
        run_main(monkeypatch, tmp_path, late, "--service-day", "2026-11-25", "--require-on-time")
    assert "not on time" in str(stop.value.code) and "only 78 of the 90" in str(stop.value.code)


def test_main_fails_with_the_reason(monkeypatch, tmp_path):
    # The exit code says that the run failed; the text says why.
    with pytest.raises(SystemExit) as stop:
        run_main(monkeypatch, tmp_path, table_until("2026-11-22"), "--service-day", "2026-11-25")
    assert "3 days before the service evening S 25.11.2026" in str(stop.value.code)


def test_main_takes_the_service_evening_from_the_clock(monkeypatch, tmp_path, capsys):
    # A fake clock at 02:05 UTC on 26.11.2026, when the run of 25.11 really committed.
    class FakeClock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime(2026, 11, 26, 2, 5, tzinfo=tz)

    monkeypatch.setattr(check_fresh.dt, "datetime", FakeClock)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    run_main(monkeypatch, tmp_path, table_until("2026-11-25"))

    assert "service evening S 25.11.2026" in capsys.readouterr().out


def test_the_daily_job_runs_the_check_after_storing_the_data():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    store = workflow.index("- name: Store the new data version")
    check = workflow.index("- name: Check that the data is fresh")
    # After storing: the newest files are saved even if the check fails.
    assert store < check
    assert "lakegeneva.features.check_fresh" in workflow[check:]
