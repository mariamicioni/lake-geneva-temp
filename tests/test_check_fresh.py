"""Tests for check_fresh.py: the daily job turns red if the feature table is too old."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push.
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

from lakegeneva.common import features
from lakegeneva.features import check_fresh

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "feature-pipeline.yml"
TODAY = dt.date(2026, 11, 1)


def table_until(last_day: str) -> pd.DataFrame:
    """A small table with all features filled, three days up to last_day."""
    days = pd.date_range(end=last_day, periods=3, freq="D", name="date")
    return pd.DataFrame(1.0, index=days, columns=features.FEATURES)


def run_main(monkeypatch, tmp_path, table, now) -> None:
    """Save the table as Parquet and start main() as if today were the date `now`."""
    path = tmp_path / "features.parquet"
    table.to_parquet(path)
    monkeypatch.setattr(sys, "argv", ["check_fresh", "--table", str(path)])

    # A fake clock, so that the test gives the same result on every day.
    class FakeClock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime(now.year, now.month, now.day, 12, tzinfo=tz)

    monkeypatch.setattr(check_fresh.dt, "datetime", FakeClock)
    check_fresh.main()


def test_age_counts_from_the_newest_row_with_all_features():
    table = table_until("2026-10-31")
    assert check_fresh.age_in_days(table, TODAY) == 1

    # The water features of the newest day exist, but its forecast is missing: this
    # row does not count, so the check also notices a forecast that stopped coming.
    table.loc[pd.Timestamp("2026-10-31"), "wind_sw_fc_d1"] = None
    assert check_fresh.age_in_days(table, TODAY) == 2


def test_a_table_without_any_complete_row_has_no_age():
    table = table_until("2026-10-31")
    table["wind_sw_fc_d1"] = None
    assert check_fresh.age_in_days(table, TODAY) is None


def test_two_days_are_allowed_and_three_days_fail(monkeypatch, tmp_path, capsys):
    # 30.10 -> 01.11 = 2 days: a late or missed run is no alarm.
    run_main(monkeypatch, tmp_path, table_until("2026-10-30"), TODAY)
    assert "2 days old" in capsys.readouterr().out

    # 29.10 -> 01.11 = 3 days: the exit code says that the run failed.
    with pytest.raises(SystemExit) as stop:
        run_main(monkeypatch, tmp_path, table_until("2026-10-29"), TODAY)
    assert "3 days old" in str(stop.value.code)


def test_the_daily_job_runs_the_check_after_storing_the_data():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    store = workflow.index("- name: Store the new data version")
    check = workflow.index("- name: Check that the data is fresh")
    # After storing: the newest files are saved even if the check fails.
    assert store < check
    assert "lakegeneva.features.check_fresh" in workflow[check:]
