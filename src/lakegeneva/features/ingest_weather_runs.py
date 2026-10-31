"""Download past weather forecasts for Geneva: what the forecast said 48 hours before.

Source: the Open-Meteo Previous Runs API (no key). For every hour it returns the
value that the forecast had two days earlier (named ..._previous_day2). The same
script is used for the daily run and for the backfill, so training and the live
forecast get exactly the same kind of weather data:

    python src/lakegeneva/features/ingest_weather_runs.py                       # last 7 days + tomorrow
    python src/lakegeneva/features/ingest_weather_runs.py --start 2024-01-20    # backfill

Uses only the standard library and one small helper of this project (no extra packages).
"""
# Why a 48-hour-old forecast: at 18:00 the model may only use what was known
# before 18:00 ("point-in-time correctness", Feature Store slides, p.9-10). The
# forecast for an hour of tomorrow that was made 48 hours earlier was made
# yesterday at the latest, so it always existed before the cutoff, however late
# GitHub starts the job (in October 2026 the runs started between 19:33 and 00:36
# Swiss time). Training on the same 48-hour-old values avoids training-serving skew.
# Checked on 21.10 and 27.10.2026: air temperature exists from 2022, sunshine and
# wind are complete from 20.01.2024 13:00, and the whole range comes in one request
# (about 17 seconds). ingest_weather.py keeps saving the newest forecast as an
# audit copy.

# All imports except the last one are from the Python standard library.
import argparse  # reads the command-line options --start, --end and --out
import csv  # writes the result as a CSV file
import datetime as dt  # dates and the time of the download
import json  # reads the API answer
import time  # time.sleep() to wait before a retry
import urllib.request  # sends the HTTP request to the API
from pathlib import Path  # file paths that work on Windows and Linux

from lakegeneva.common import raw_files  # a second run never replaces an earlier file

# --- Settings -----------------------------------------------------------------
API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
# Geneva, the same point as in ingest_weather.py (see config/settings.yaml).
LATITUDE = 46.20
LONGITUDE = 6.14
# The same four hourly variables as in ingest_weather.py (explained there).
VARIABLES = ["temperature_2m", "sunshine_duration", "wind_speed_10m", "wind_direction_10m"]
# The API names the 48-hour-old values with this ending, e.g.
# temperature_2m_previous_day2. We keep these names in the CSV, so the file itself
# says that it holds the old forecast and not the newest one.
SUFFIX = "_previous_day2"
COLUMNS = [variable + SUFFIX for variable in VARIABLES]
# Polite identification: tells the API owner who is calling.
USER_AGENT = "lake-geneva-temp (student project, github.com/mariamicioni/lake-geneva-temp)"


def fetch(start: dt.date, end: dt.date) -> dict:
    """Return the hourly 48-hour-old forecast from start to end (both days included)."""
    # REST API: everything we want is in the URL.
    # timezone=GMT: the times come back in UTC, like the water readings. (With
    # timezone=Europe/Zurich, Open-Meteo shifts ALL hours by the offset that is valid
    # at the moment of the request - +1 h in winter even for summer days, checked on
    # 27.10.2026.) features.py converts UTC to Swiss time, with summer time.
    url = (
        f"{API_URL}?latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&hourly={','.join(COLUMNS)}&start_date={start}&end_date={end}&timezone=GMT"
    )
    # Try a few times in case the API is briefly unavailable.
    for attempt in range(1, 4):  # attempts 1, 2 and 3
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            # timeout=120: a backfill of almost three years is one big answer.
            with urllib.request.urlopen(request, timeout=120) as response:
                # "hourly" holds a "time" list plus one list per column, all the same
                # length: hourly["temperature_2m_previous_day2"][i] belongs to hourly["time"][i].
                return json.load(response)["hourly"]
        except (OSError, ValueError, KeyError) as error:
            # OSError: network problem or timeout; ValueError: invalid JSON;
            # KeyError: the answer has no "hourly" part.
            if attempt == 3:
                raise  # third failure: give up and let the job fail visibly
            print(f"Attempt {attempt} failed ({error}), retrying in 60 s")
            time.sleep(60)


def write_csv(hourly: dict, fetched_at: dt.datetime, path: Path) -> None:
    # Create the output folder if it does not exist yet (no error if it does).
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" stops Windows from writing an empty line after every row.
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fetched_at_utc", "time_utc", *COLUMNS])
        # One row per hour (in UTC). A missing value (None, e.g. sunshine before
        # 20.01.2024) becomes an empty cell, so it stays visibly empty instead of 0.
        for i, time_utc in enumerate(hourly["time"]):
            writer.writerow([fetched_at.isoformat(timespec="seconds"), time_utc, *(hourly[c][i] for c in COLUMNS)])


def main() -> None:
    # Command-line options; __doc__ (the text at the top) becomes the --help text.
    # RawDescriptionHelpFormatter keeps the line breaks of the examples in --help.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # type=dt.date.fromisoformat turns "2024-01-20" into a date object.
    parser.add_argument("--start", type=dt.date.fromisoformat, help="first day, YYYY-MM-DD (default: 7 days ago)")
    parser.add_argument("--end", type=dt.date.fromisoformat, help="last day, YYYY-MM-DD (default: tomorrow)")
    parser.add_argument("--out", default="data/raw", help="output folder")
    args = parser.parse_args()

    # The download time, saved in every row: if two files hold the same hour, the
    # features can take the value from the newest file.
    now = dt.datetime.now(dt.timezone.utc)
    # Defaults for the daily run: tomorrow (the day we predict) and the 7 days
    # before today, as UTC days (tomorrow's Swiss day ends at 22:00 or 23:00 UTC,
    # so it is inside). A 48-hour-old forecast is never changed afterwards, so the
    # overlap with earlier runs is harmless and fills in a missed day.
    end = args.end or now.date() + dt.timedelta(days=1)
    start = args.start or now.date() - dt.timedelta(days=7)
    hourly = fetch(start, end)
    # An answer with no hours, or with only empty values, must not become a file: the
    # job would stay green although the API gave no data, and in a run without older
    # files (the Docker image) an empty file makes the feature build fail with a
    # confusing error. SystemExit stops the script, so the job turns red.
    if not any(value is not None for column in COLUMNS for value in hourly.get(column, [])):
        raise SystemExit("The API returned no forecast values.")

    # One file per run, named after the range, e.g. weather_geneva_prev2_2024-01-20_to_2026-10-28.csv
    # A second run with the same range gets its own file (see common/raw_files.py).
    path = raw_files.new_path(Path(args.out) / f"weather_geneva_prev2_{start}_to_{end}.csv")
    write_csv(hourly, now, path)
    print(f"Saved {len(hourly['time'])} forecast hours ({start} to {end}) to {path}")


# Runs main() only when the file is started as a script, not when it is imported
# (the tests import this file to test single functions).
if __name__ == "__main__":
    main()
