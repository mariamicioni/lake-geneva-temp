"""Feature pipeline, last step: all raw files -> the daily feature table -> Parquet.

The table is computed with the shared definition in common/features.py and saved
as data/features/features.parquet: one row per day, all features and the label.
This file is the feature store. DVC versions it and keeps it in the GCS bucket.

    python src/lakegeneva/features/build_features.py
"""
# Why Parquet and not CSV: Parquet stores the column types (dates stay dates,
# numbers stay numbers) and is compressed, so the training pipeline reads back
# exactly the table that was written. The course FAQ accepts versioned Parquet
# files on GCS as a "poor man's feature store" (Feature Store slides).
# Why always the whole table: it takes about 2 seconds, and recomputing everything
# from the raw files means the stored table can never drift away from the raw
# data or from the feature definition.

import argparse  # reads the command-line options --raw and --out
from pathlib import Path  # file paths that work on Windows and Linux

import pandas as pd  # writes the table as Parquet (with pyarrow)

from lakegeneva.common import features  # the one shared feature definition


def build(raw_dir: Path | str = "data/raw", out: Path | str = "data/features/features.parquet") -> pd.DataFrame:
    """Compute the feature table from the raw files and save it as Parquet."""
    table = features.feature_table(raw_dir)
    out = Path(out)
    # Create the output folder if it does not exist yet (no error if it does).
    out.parent.mkdir(parents=True, exist_ok=True)
    # The dates (the index) are saved too, so the file can be read back as it was.
    table.to_parquet(out)
    return table


def main() -> None:
    # Command-line options; __doc__ (the text at the top) becomes the --help text.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default="data/raw", help="folder with the raw files")
    parser.add_argument("--out", default="data/features/features.parquet", help="where to save the table")
    args = parser.parse_args()

    table = build(args.raw, args.out)
    # Rows the model can learn from: every feature and the label exist.
    complete = table[features.FEATURES + [features.LABEL]].notna().all(axis=1)
    print(
        f"Saved {len(table)} days ({table.index.min():%d.%m.%Y} to {table.index.max():%d.%m.%Y}) "
        f"to {args.out}; {int(complete.sum())} days have all features and a label"
    )


# Runs main() only when the file is started as a script, not when it is imported
# (the tests import this file to test build()).
if __name__ == "__main__":
    main()
