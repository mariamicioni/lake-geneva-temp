"""Save the current Open-Meteo weather forecast for Geneva.

Each run writes one CSV snapshot: the hourly forecast for today and the next 2 days,
as it was published at the time of the run. This keeps a copy of exactly what the
forecast said, next to the past forecasts that the previous-runs API can return later.
Only uses the standard library, so it runs without installing anything.
"""
# Why we save forecasts and not measured weather: at 18:00 the model can only know
# tomorrow's weather as a FORECAST. Training on measured weather would give the
# model perfect weather knowledge it never has in production (leakage). Saving each
# day's forecast at the time of the run keeps exactly what was known then
# ("point-in-time correctness", Feature Store slides, p.9-10).

# All imports are from the Python standard library (nothing to install).
import argparse  # reads the command-line option --out
import csv  # writes the result as a CSV file
import datetime as dt  # the time of the download
import json  # reads the API answer
import time  # time.sleep() to wait before a retry
import urllib.request  # sends the HTTP request to the API
from pathlib import Path  # file paths that work on Windows and Linux

# --- Settings -----------------------------------------------------------------
API_URL = "https://api.open-meteo.com/v1/forecast"
# Geneva, close to the station and the Bains des Pâquis (see config/settings.yaml).
LATITUDE = 46.20
LONGITUDE = 6.14
# Hourly weather variables, with Open-Meteo's names:
# - temperature_2m: air temperature 2 m above ground (°C)
# - sunshine_duration: seconds of sunshine in each hour (0-3600)
# - wind_speed_10m: wind speed 10 m above ground (km/h)
# wind_direction_10m: degrees the wind comes FROM (225 = south-west). Strong
# south-westerly wind causes upwelling and sharp temperature drops at Geneva.
VARIABLES = ["temperature_2m", "sunshine_duration", "wind_speed_10m", "wind_direction_10m"]
# Polite identification: tells the API owner who is calling.
USER_AGENT = "lake-geneva-temp (student project, github.com/mariamicioni/lake-geneva-temp)"


def fetch() -> dict:
    # REST API: everything we want is in the URL.
    # forecast_days=3: today, tomorrow (the day we predict) and the day after.
    # timezone=GMT: times in UTC, like all raw data (see ingest_weather_runs.py for
    # why not Europe/Zurich). Files before 27.10.2026 have "time_local" instead:
    # Swiss clock time with the offset that was valid at the download.
    url = (
        f"{API_URL}?latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&hourly={','.join(VARIABLES)}&forecast_days=3&timezone=GMT"
    )
    # Try a few times in case the API is briefly unavailable.
    for attempt in range(1, 4):  # attempts 1, 2 and 3
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                # "hourly" holds one list per variable, plus a "time" list, all the
                # same length: hourly["temperature_2m"][i] belongs to hourly["time"][i].
                return json.load(response)["hourly"]
        except (OSError, ValueError, KeyError) as error:
            # OSError: network problem or timeout; ValueError: invalid JSON;
            # KeyError: the answer has no "hourly" part.
            if attempt == 3:
                raise  # third failure: give up and let the job fail visibly
            print(f"Attempt {attempt} failed ({error}), retrying in 60 s")
            time.sleep(60)


def main() -> None:
    # Command-line option; __doc__ (the text at the top) becomes the --help text.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw", help="output folder")
    args = parser.parse_args()

    # The download time, saved in every row: later we can prove that a forecast
    # was really available before the 18:00 cutoff.
    now = dt.datetime.now(dt.timezone.utc)
    hourly = fetch()

    # One file per run, e.g. weather_geneva_fetched_2026-10-20.csv
    path = Path(args.out) / f"weather_geneva_fetched_{now.date()}.csv"
    # Create the output folder if it does not exist yet (no error if it does).
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" stops Windows from writing an empty line after every row.
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fetched_at_utc", "time_utc", *VARIABLES])
        # One row per forecast hour (3 days x 24 hours = 72 rows), in UTC.
        for i, time_utc in enumerate(hourly["time"]):
            writer.writerow([now.isoformat(timespec="seconds"), time_utc, *(hourly[v][i] for v in VARIABLES)])
    print(f"Saved {len(hourly['time'])} forecast hours to {path}")


# Runs main() only when the file is started as a script, not when it is imported.
if __name__ == "__main__":
    main()
