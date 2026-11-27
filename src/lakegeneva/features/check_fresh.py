"""Feature pipeline, after storing: check the row of the issue day D for the evening S.

Every evening S (the service evening) the job should store the row of S itself. The
issue day D for S is the newest day up to S whose readings before 18:00 are complete
(the rule of features.issue_day, MS2 review point A1). This step checks the row of D:

- D is S: on time.
- D is the day before S: delayed. S did not have enough readings before 18:00 yet when
  the job ran, so the inputs for the forecast of S+1 are missing. The step stays green,
  but shows a warning in the summary of the run.
- No D, D older than the day before S, or a feature missing in the row of D: the job
  turns red and GitHub sends an email.

    python -m lakegeneva.features.check_fresh                            # S from the clock
    python -m lakegeneva.features.check_fresh --service-day 2026-11-25   # S given
    python -m lakegeneva.features.check_fresh --require-on-time          # only D = S passes

S from the clock is the evening of the newest scheduled start, so a run started by
hand or again the next day checks the current evening, with the data it downloaded
itself. To check an OLD data version for its own evening, get that table (dvc get, or
feature_store.load_features) and give its evening with --table and --service-day.
"""
# Why this check (MS2 review point A2): a download only fails if the API gives no
# answer. If a source kept sending old data, or a forecast stopped coming, every step
# would still end without an error and the job would stay green.
# Why the row of D and not an age in days: before, the step only asked "is the newest
# row with all features at most 2 days old?". So it stayed green when the row of D
# missed a feature, as long as an older row was complete, and it never said whether
# today's row was there. Now it names S and D and looks at exactly the row of D.
# Why a one-day delay is still green: in the 16 evenings from 11.11 to 26.11.2026, S
# had only 66 to 84 of the 90 readings before 18:00 when the job ran on 6 evenings; on
# the other 10 D was S. All 6 days reached 108 readings in a later download, so the
# readings came late. A red job on so many nights would only say that, and the next
# night downloads the missing readings again. For the live forecast in MS4, an on-time
# forecast needs D = S (MAX_DELAY_DAYS = 0 there, with a second run later in the
# evening, review point E4).
# Why a separate step after the data is stored, and not inside build_features: the
# newest files are saved even if the check fails. And an older data version can still
# be rebuilt, for example with the Docker command in the README.

import argparse  # reads the command-line options --table and --service-day
import datetime as dt  # the clock and the dates
import os  # os.environ: is this a run on GitHub?

import numpy as np  # np.isfinite: a feature that is a real number
import pandas as pd  # the type of the table (feature_store.py reads it)

from lakegeneva.common import feature_store, features  # the features and the issue day; the read path

# How many days the issue day D may be before the service evening S and still pass.
MAX_DELAY_DAYS = 1
# The scheduled start of the daily job: 19:00 UTC (feature-pipeline.yml, cron
# "0 19 * * *"), which is 20:00 or 21:00 in Switzerland, after the 18:00 cutoff.
SCHEDULE_HOUR_UTC = 19


def service_day(now: dt.datetime) -> dt.date:
    """The service evening S: the day of the newest scheduled start of the daily job at or before `now`."""
    # GitHub starts the evening job hours late, often after midnight (for the evenings
    # 11.11 to 25.11.2026 the bot committed between 00:01 and 05:16 Swiss time). The calendar
    # date of the run would then be the next day. The newest scheduled start (19:00
    # UTC) that has passed names the evening the run is for, whatever its real start
    # time: 02:05 UTC on 26.11 -> S = 25.11; 19:30 UTC on 25.11 -> 25.11; 18:30 UTC on
    # 25.11 -> 24.11 (the job of 25.11 is not due yet).
    # It also fits the Monday retrain, which GitHub started as late as 16:44 UTC: its
    # S is Sunday, whose job has run. (A first version used "the newest 18:00 Swiss
    # time" instead. In summer that is 16:00 UTC, so a late Monday retrain would have
    # asked for Monday's row before Monday's job could even start; found in a check on
    # 26.11.2026.) At 19:00 UTC Switzerland has the same date, so S is a Swiss date too.
    # `now` must know its time zone (e.g. dt.timezone.utc).
    utc = now.astimezone(dt.UTC)
    if utc.hour >= SCHEDULE_HOUR_UTC:
        return utc.date()
    return utc.date() - dt.timedelta(days=1)


def readings_before_cutoff(table: pd.DataFrame, day: dt.date) -> int:
    """How many water readings before 18:00 the table has for `day` (0: no row, or none)."""
    value = table["n_readings_d0"].get(pd.Timestamp(day))
    return 0 if value is None or pd.isna(value) else int(value)


def issue_day_for(table: pd.DataFrame, service: dt.date) -> pd.Timestamp | None:
    """The issue day D for the evening S: the newest day UP TO S whose readings before 18:00 are complete."""
    # The same rule as features.issue_day (A1), but rows after S do not count: the job
    # of evening S needs the row of S (on time) or of the day before (delayed). A newer
    # complete day says nothing about the row of S. Later rows may exist: a row of
    # tomorrow with the weather only, or a newer day with readings when the check runs
    # later (found in a review on 27.11.2026: the row of S missed the wind, but a
    # complete newer day made the check say "on time").
    return features.issue_day(table[table.index <= pd.Timestamp(service)])


def check_issue_day(table: pd.DataFrame, service: dt.date) -> tuple[bool, bool, str]:
    """Is the row of the issue day D ready for the service evening S? Returns (passed, delayed, text)."""
    # One row per whole date (midnight): with two rows for one day, or a row at another
    # time of the day, "the row of D" would not be clear, and its count of readings
    # could be read as 0. build_features makes one row per day at midnight, so this only
    # stops a broken table (review of 27.11.2026).
    dates = table.index
    if not dates.is_unique or not (dates == dates.normalize()).all():
        return False, False, "the table must have exactly one row per date, at midnight"
    # A count of readings is a whole number. An infinite one would pass "at least 90"
    # and then break the message (review of 27.11.2026).
    if np.isinf(table["n_readings_d0"].to_numpy(dtype=float)).any():
        return False, False, "a count of readings before 18:00 is infinite"
    day = issue_day_for(table, service)
    if day is None:
        return False, False, (
            f"no day up to the service evening S {service:%d.%m.%Y} has {features.MIN_READINGS_D0} "
            f"readings before 18:00, so there is no issue day D"
        )
    issue = day.date()
    # How many days D is behind S (0 or more: D is never after S).
    delay = (service - issue).days
    if delay > MAX_DELAY_DAYS:
        return False, False, (
            f"issue day D {issue:%d.%m.%Y} is {delay} days before the service evening S "
            f"{service:%d.%m.%Y} (allowed: {MAX_DELAY_DAYS})"
        )
    # The row of D itself must have every feature, also the weather of the forecast.
    # The old age rule passed when this row missed a feature but an older row was
    # complete. np.isfinite: no missing value (NaN) and no infinity.
    values = table.loc[day, features.FEATURES].to_numpy(dtype=float)
    missing = [column for column, value in zip(features.FEATURES, values, strict=True) if not np.isfinite(value)]
    if missing:
        return False, False, f"the row of issue day D {issue:%d.%m.%Y} misses {', '.join(missing)}"
    if delay == 0:
        return True, False, (
            f"on time: issue day D {issue:%d.%m.%Y}, service evening S {service:%d.%m.%Y}, all "
            f"{len(features.FEATURES)} features ({readings_before_cutoff(table, issue)} readings before 18:00)"
        )
    # The row of a day forecasts the NEXT day. So when the row of S is not ready, the
    # inputs for the forecast of S+1 are missing, and the newest ready row (D = S-1)
    # only forecasts S itself (review of 27.11.2026: a first text said the forecast of
    # S+1 "would come from the row of D", which the model cannot do). The text says only
    # what the table shows: the count of readings, not why they were missing.
    target = service + dt.timedelta(days=1)
    return True, True, (
        f"delayed: the service evening S {service:%d.%m.%Y} had only "
        f"{readings_before_cutoff(table, service)} of the {features.MIN_READINGS_D0} readings before "
        f"18:00 that it needs, so the inputs for the forecast of {target:%d.%m.%Y} are missing. The newest "
        f"ready row is the issue day D {issue:%d.%m.%Y}, which forecasts {service:%d.%m.%Y}"
    )


def main() -> None:
    # Command-line options; __doc__ (the text at the top) becomes the --help text.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", default="data/features/features.parquet", help="the feature table")
    # type=dt.date.fromisoformat turns "2026-11-25" into a date object.
    parser.add_argument(
        "--service-day", type=dt.date.fromisoformat, help="the service evening S, YYYY-MM-DD (default: from the clock)"
    )
    # Only D = S passes: the question of the catch-up run (feature-pipeline.yml, MS2
    # review point E4) and later of the live forecast in MS4, which needs the row of S.
    parser.add_argument(
        "--require-on-time", action="store_true", help="fail also when D is the day before S (delayed)"
    )
    args = parser.parse_args()

    service = args.service_day or service_day(dt.datetime.now(dt.UTC))
    # The one read path of the feature table (feature_store.py, review point E1).
    table, _ = feature_store.load_features(path=args.table)
    passed, delayed, text = check_issue_day(table, service)
    # SystemExit with a text: Python prints it and the exit code says that the run failed.
    if not passed:
        raise SystemExit(f"{args.table}: {text}.")
    if delayed and args.require_on_time:
        raise SystemExit(f"{args.table}: not on time ({text}).")
    # On GitHub, a line in this form becomes a yellow warning in the summary of the run,
    # without failing it (GitHub Actions "workflow commands").
    if delayed and os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::warning title=Issue day delayed::{text}.")
    print(f"{text}.")


# Runs main() only when the file is started as a script, not when it is imported
# (the tests import this file).
if __name__ == "__main__":
    main()
