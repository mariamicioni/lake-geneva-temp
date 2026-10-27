"""Download the forecast of the last weather model run published before 18:00.

Source: the Open-Meteo Single Runs API (no key). For every issue day D it asks for
the ECMWF IFS run that started at 06:00 UTC on D. ECMWF needs about 4-6 hours to
publish a run, so this one is ready by about 12:00 UTC: always before the 18:00
cutoff in Geneva (16:00 UTC in summer time, 17:00 UTC in winter time). The 12:00
UTC run is published too late on some days. The same script is used for the daily
run and for the backfill:

    python src/lakegeneva/features/ingest_weather_single.py                       # last 7 days
    python src/lakegeneva/features/ingest_weather_single.py --start 2024-08-10    # backfill

Only uses the standard library, so it runs without installing anything.
"""
# Why a second weather source: the features use the 48-hour-old forecast
# (ingest_weather_runs.py), which is always safe but old. This run is about one day
# fresher and still published before 18:00 - the plan of the proposal ("a run that
# was actually available by 18:00, accounting for model publication delay"). The
# training pipeline (MS3) can then compare the two. Note: it is also another model
# (ECMWF IFS instead of Open-Meteo's "best match"), so that comparison mixes
# "fresher" and "other model".
# Archive checked on 27.10.2026: 06:00 UTC runs exist from 10.08.2024 (the 00:00 and
# 12:00 UTC runs from 14.03.2024). Single days are missing (e.g. 05.08.2025); they
# are skipped, not treated as an error.

# All imports are from the Python standard library (nothing to install).
import argparse  # reads the command-line options --start, --end and --out
import csv  # writes the result as a CSV file
import datetime as dt  # dates and the time of the download
import json  # reads the API answer
import time  # time.sleep() between requests and before a retry
import urllib.error  # HTTPError: the API answered with an error code
import urllib.request  # sends the HTTP request to the API
from pathlib import Path  # file paths that work on Windows and Linux

# --- Settings -----------------------------------------------------------------
API_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
# Geneva, the same point as in ingest_weather.py (see config/settings.yaml).
LATITUDE = 46.20
LONGITUDE = 6.14
# The same four hourly variables as in the other weather scripts.
VARIABLES = ["temperature_2m", "sunshine_duration", "wind_speed_10m", "wind_direction_10m"]
# ECMWF IFS HRES (9 km), the model with the longest archive of single runs.
MODEL = "ecmwf_ifs"
# The run that starts at 06:00 UTC: the last one that is surely published by 18:00.
RUN_HOUR = 6
# 48 hours from 06:00 UTC on D reach 06:00 UTC on D+2, so every hour of tomorrow
# (00:00 to 23:00 Swiss time) is included.
FORECAST_HOURS = 48
# Polite identification: tells the API owner who is calling.
USER_AGENT = "lake-geneva-temp (student project, github.com/mariamicioni/lake-geneva-temp)"


def fetch_run(day: dt.date) -> dict | None:
    """The hourly forecast of the 06:00 UTC run of `day`, or None if that run is not archived."""
    # REST API: everything we want is in the URL. run= is the start of the run in
    # UTC; timezone=GMT gives the forecast hours in UTC too (see ingest_weather_runs.py
    # for why not Europe/Zurich).
    url = (
        f"{API_URL}?latitude={LATITUDE}&longitude={LONGITUDE}&hourly={','.join(VARIABLES)}"
        f"&models={MODEL}&run={day}T{RUN_HOUR:02d}:00&forecast_hours={FORECAST_HOURS}&timezone=GMT"
    )
    for attempt in range(1, 4):  # attempts 1, 2 and 3
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                # "hourly" holds a "time" list plus one list per variable, all the same length.
                return json.load(response)["hourly"]
        except urllib.error.HTTPError as error:
            # HTTP 400 "The requested model run is not available": a day missing in
            # the archive, or today's run before it is published. Not a problem of
            # the job: this day is simply skipped.
            if error.code == 400 and b"not available" in error.read():
                return None
            if attempt == 3:
                raise  # another error code three times: let the job fail visibly
            print(f"Attempt {attempt} for {day} failed ({error}), retrying in 60 s")
            time.sleep(60)
        except (OSError, ValueError, KeyError) as error:
            # OSError: network problem or timeout; ValueError: invalid JSON;
            # KeyError: the answer has no "hourly" part.
            if attempt == 3:
                raise
            print(f"Attempt {attempt} for {day} failed ({error}), retrying in 60 s")
            time.sleep(60)


def write_csv(runs: dict[dt.date, dict], fetched_at: dt.datetime, path: Path) -> None:
    # Create the output folder if it does not exist yet (no error if it does).
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" stops Windows from writing an empty line after every row.
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fetched_at_utc", "run_utc", "time_utc", *VARIABLES])
        # One row per run and forecast hour (in UTC). run_utc says which run a value
        # comes from; a missing value (None) becomes an empty cell, not 0.
        for day in sorted(runs):
            hourly = runs[day]
            run_utc = f"{day}T{RUN_HOUR:02d}:00Z"
            for i, time_utc in enumerate(hourly["time"]):
                writer.writerow(
                    [fetched_at.isoformat(timespec="seconds"), run_utc, time_utc, *(hourly[v][i] for v in VARIABLES)]
                )


def main() -> None:
    # Command-line options; __doc__ (the text at the top) becomes the --help text.
    # RawDescriptionHelpFormatter keeps the line breaks of the examples in --help.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # type=dt.date.fromisoformat turns "2024-08-10" into a date object.
    parser.add_argument("--start", type=dt.date.fromisoformat, help="first issue day, YYYY-MM-DD (default: 6 days before --end)")
    parser.add_argument("--end", type=dt.date.fromisoformat, help="last issue day, YYYY-MM-DD (default: today)")
    parser.add_argument("--out", default="data/raw", help="output folder")
    args = parser.parse_args()

    # The download time, saved in every row.
    now = dt.datetime.now(dt.timezone.utc)
    # Defaults for the daily run: the runs of today and the 6 days before (UTC
    # days). The archive does not change, so the overlap with earlier runs is
    # harmless and fills in a missed day.
    end = args.end or now.date()
    start = args.start or end - dt.timedelta(days=6)

    runs: dict[dt.date, dict] = {}
    skipped = []
    day = start
    while day <= end:
        hourly = fetch_run(day)
        if hourly is None:
            skipped.append(str(day))
        else:
            runs[day] = hourly
        day += dt.timedelta(days=1)
        time.sleep(0.2)  # a short pause between requests, to be polite to the API
    if skipped:
        print(f"No 06:00 UTC run in the archive for {len(skipped)} day(s): {', '.join(skipped)}")
    if not runs:
        # SystemExit stops the script with an error, so the GitHub job turns red.
        raise SystemExit("The API returned no run at all.")

    # One file per run of the script, e.g. weather_geneva_single06_2024-08-10_to_2026-10-27.csv
    path = Path(args.out) / f"weather_geneva_single06_{start}_to_{end}.csv"
    write_csv(runs, now, path)
    print(f"Saved {len(runs)} runs ({min(runs)} to {max(runs)}) to {path}")


# Runs main() only when the file is started as a script, not when it is imported
# (the tests import this file to test single functions).
if __name__ == "__main__":
    main()
