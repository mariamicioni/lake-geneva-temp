"""Tests for run_pipeline.py: all steps of the feature pipeline, in the right order."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push.
# No test here downloads anything: each step's main() is replaced by a small fake.
# The real steps have their own tests in the other test files.
import re
import sys
from pathlib import Path

import pytest

from lakegeneva.features import (
    build_features,
    ingest_water,
    ingest_weather,
    ingest_weather_runs,
    ingest_weather_single,
    run_pipeline,
)

ALL_STEPS = [ingest_water, ingest_weather_runs, ingest_weather, ingest_weather_single, build_features]
WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "feature-pipeline.yml"


def record_steps(monkeypatch) -> list:
    """Replace each step's main() by a fake that only writes down that it ran."""
    # monkeypatch changes something for one test and puts it back afterwards.
    ran = []
    for step in ALL_STEPS:
        # step=step gives each fake its own step (otherwise all would see the last one).
        monkeypatch.setattr(step, "main", lambda step=step: ran.append(step))
    # The command line of a normal start: the program name and no options.
    monkeypatch.setattr(sys, "argv", ["run_pipeline"])
    return ran


def test_all_steps_run_and_the_table_comes_last(monkeypatch):
    ran = record_steps(monkeypatch)

    run_pipeline.main()

    # Every step ran exactly once, and the table is built after all the downloads.
    assert sorted(ran, key=ALL_STEPS.index) == ALL_STEPS
    assert ran[-1] is build_features


def test_a_failed_step_stops_the_run(monkeypatch):
    ran = record_steps(monkeypatch)

    def fail():
        raise OSError("the API does not answer")

    # The second step fails, after the water download has worked.
    monkeypatch.setattr(ingest_weather_runs, "main", fail)

    # pytest.raises: the test passes only if the error comes out of main(). This is
    # what makes "docker run" end with an error code.
    with pytest.raises(OSError):
        run_pipeline.main()
    # Only the water download ran: no later step, and no table from incomplete data.
    assert ran == [ingest_water]


def test_help_stops_before_anything_is_downloaded(monkeypatch):
    # CI starts "run_pipeline --help" in the image: it must not call the APIs.
    ran = record_steps(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_pipeline", "--help"])

    # argparse prints the help text and ends the program with exit code 0 (no error).
    with pytest.raises(SystemExit) as stop:
        run_pipeline.main()

    assert stop.value.code == 0
    assert ran == []


def test_the_daily_job_has_the_same_steps_in_the_same_order():
    # The daily job lists its steps in the workflow file, this file lists them in
    # STEPS. Read the script names from the workflow, e.g. "ingest_water".
    in_workflow = re.findall(r"src/lakegeneva/features/(\w+)\.py", WORKFLOW.read_text(encoding="utf-8"))
    # step.__name__ is the full name "lakegeneva.features.ingest_water"; keep the last part.
    in_steps = [step.__name__.split(".")[-1] for step in run_pipeline.STEPS]

    assert in_workflow == in_steps
