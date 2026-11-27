"""Check the feature table that the pipeline built from the saved answers of this folder.

Run inside the Docker image by CI (.github/workflows/ci.yml) and by tests/test_replay.py,
so both apply exactly the same rules (MS2 review point A3):

    python check_table.py data/features/features.parquet
"""
# Why more than the issue day: the issue day depends only on the count of water
# readings. A review on 27.11.2026 removed the wind from a saved answer: the date was
# still right, but only 5 of the 10 features of that row were usable, and the check of
# the date alone stayed green. So the row of D must have every feature, and some rows
# must have a label.

import sys  # the path of the table from the command line

import numpy as np  # np.isfinite: a real number, not NaN and not infinite
import pandas as pd  # reads the table

from lakegeneva.common import features  # the feature names, the label and the issue day

# The saved answers are from 20.-27.11.2026: 26.11 is the newest day with at least 90
# readings before 18:00.
EXPECTED_ISSUE_DAY = pd.Timestamp("2026-11-26")


def check(table: pd.DataFrame) -> str:
    """Stop with an error unless the table is complete for the saved answers; returns a short report."""
    day = features.issue_day(table)
    if day != EXPECTED_ISSUE_DAY:
        raise SystemExit(f"issue day {day}, expected {EXPECTED_ISSUE_DAY:%Y-%m-%d}")
    values = table.loc[day, features.FEATURES].to_numpy(dtype=float)
    missing = [name for name, value in zip(features.FEATURES, values, strict=True) if not np.isfinite(value)]
    if missing:
        raise SystemExit(f"the row of {day:%Y-%m-%d} misses {', '.join(missing)}")
    labelled = int(table[features.LABEL].notna().sum())
    if labelled == 0:
        raise SystemExit("no row has a label")
    return f"table OK: issue day {day:%Y-%m-%d} with all {len(features.FEATURES)} features, {labelled} rows with a label"


if __name__ == "__main__":
    print(check(pd.read_parquet(sys.argv[1])))
