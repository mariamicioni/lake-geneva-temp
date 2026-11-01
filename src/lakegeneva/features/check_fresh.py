"""Feature pipeline, after storing: stop with an error if the feature table is too old.

Every day should add a row with all features. If the newest such row is more than
2 days old, this step fails, so the daily job turns red and GitHub sends an email.

    python -m lakegeneva.features.check_fresh
"""
# Why this check: a download only fails if the API gives no answer. If a source kept
# sending old data, or a forecast stopped coming, every step would still end without
# an error and the job would stay green.
# Why 2 days: GitHub often starts the job late, sometimes after midnight, and one
# missed day does no harm, because the next run downloads the last 7 days again.
# Why a separate step after the data is stored, and not inside build_features: the
# newest files are saved even if the check fails. And an older data version can still
# be rebuilt, for example with the Docker command in the README.

import argparse  # reads the command-line option --table
import datetime as dt  # today's date
from zoneinfo import ZoneInfo  # the Swiss clock

import pandas as pd  # reads the Parquet table

from lakegeneva.common import features  # the list of features and the time zone

# The newest row with all features may be at most this many days old.
MAX_AGE_DAYS = 2


def age_in_days(table: pd.DataFrame, today: dt.date) -> int | None:
    """Days between today and the newest row with all features (None: no such row)."""
    complete = table[features.FEATURES].notna().all(axis=1)
    if not complete.any():
        return None
    # complete[complete] keeps only the True rows; the index holds their dates.
    return (today - complete[complete].index.max().date()).days


def main() -> None:
    # Command-line options; __doc__ (the text at the top) becomes the --help text.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", default="data/features/features.parquet", help="the feature table")
    args = parser.parse_args()

    today = dt.datetime.now(ZoneInfo(features.TIMEZONE)).date()
    age = age_in_days(pd.read_parquet(args.table), today)
    # SystemExit with a text: Python prints it and the exit code says that the run failed.
    if age is None:
        raise SystemExit(f"{args.table} has no row with all features.")
    if age > MAX_AGE_DAYS:
        raise SystemExit(f"The newest row with all features is {age} days old (allowed: {MAX_AGE_DAYS}).")
    print(f"The newest row with all features is {age} days old (allowed: {MAX_AGE_DAYS}).")


# Runs main() only when the file is started as a script, not when it is imported
# (the tests import this file).
if __name__ == "__main__":
    main()
