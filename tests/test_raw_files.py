"""Tests for the rule that a raw file is never overwritten. They do not use the internet."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push
# (.github/workflows/ci.yml). pytest finds every function whose name starts with
# test_ and runs it; a test fails if one of its `assert` lines is false.
# Each download script also has its own test that runs the script twice.
from lakegeneva.common import raw_files


def test_a_free_name_stays_as_it_is(tmp_path):
    # tmp_path (a pytest helper) is a fresh temporary folder, deleted afterwards.
    path = tmp_path / "water_2606_fetched_2026-10-31.csv"

    assert raw_files.new_path(path) == path


def test_a_taken_name_gets_the_next_free_number(tmp_path):
    path = tmp_path / "water_2606_fetched_2026-10-31.csv"
    path.write_text("first run")
    assert raw_files.new_path(path).name == "water_2606_fetched_2026-10-31_run02.csv"

    (tmp_path / "water_2606_fetched_2026-10-31_run02.csv").write_text("second run")
    assert raw_files.new_path(path).name == "water_2606_fetched_2026-10-31_run03.csv"


def test_the_name_order_stays_the_download_order(tmp_path):
    # load_water() reads the files in name order and the last one wins, so the order
    # of the names must be the order of the downloads, also after run 9.
    first = tmp_path / "water_2606_fetched_2026-10-31.csv"
    created = []
    for _ in range(11):  # the normal name, then _run02 to _run11
        path = raw_files.new_path(first)
        path.write_text("one run")
        created.append(path)
    # The file of the next day comes after all the runs of this day.
    next_day = tmp_path / "water_2606_fetched_2026-11-01.csv"
    next_day.write_text("next day")

    assert sorted(tmp_path.glob("*.csv")) == [*created, next_day]
