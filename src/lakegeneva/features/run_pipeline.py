"""Feature pipeline, all steps at once: the newest data -> the daily feature table.

Runs the four downloads and then builds the table, in the same order as the daily
job (.github/workflows/feature-pipeline.yml). The downloads use public APIs, so no
key is needed. The Docker image starts this file by default:

    python -m lakegeneva.features.run_pipeline
"""
# Why a Python file and not a shell command in the Dockerfile: Docker starts one
# program directly, with no shell in between ("no shell tricks", Block 3 exercise
# [12] in README.md), and this file can be tested like the rest of the code.
# Why the daily job does not use this file: it shows each step separately in the
# Actions log, and its weather downloads still run if the water download failed.
# Here the first error stops the run, which is what a quick check needs.

import argparse  # only for --help: this file has no other options

# The five steps. Each one has its own main() function.
from lakegeneva.features import (
    build_features,
    ingest_water,
    ingest_weather,
    ingest_weather_runs,
    ingest_weather_single,
)

# The same order as in the daily job; tests/test_run_pipeline.py checks that the two
# stay the same. The table comes last: it is built from the files just downloaded.
STEPS = (ingest_water, ingest_weather_runs, ingest_weather, ingest_weather_single, build_features)


def main() -> None:
    # Command-line options: none. __doc__ (the text at the top) becomes the --help
    # text. --help or any other option ends the program here, before any download.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    # Every step reads its options from the same command line, which is empty here.
    # So all steps use their defaults: the last 7 days, the folder data/raw and the
    # table data/features/features.parquet. These paths start from the current folder.
    for step in STEPS:
        # An error is not caught here: Python stops, and the exit code says that the
        # run failed. So no table is built from incomplete downloads.
        step.main()


# Runs main() only when the file is started as a script, not when it is imported
# (the tests import this file to test main()).
if __name__ == "__main__":
    main()
