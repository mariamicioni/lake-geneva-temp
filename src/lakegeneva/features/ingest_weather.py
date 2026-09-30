"""Save the current Open-Meteo weather forecast for Geneva.

Each run writes one CSV snapshot: the hourly forecast for today and the next 2 days,
as it was published at the time of the run. This keeps a copy of exactly what the
forecast said, next to the past forecasts that the previous-runs API can return later.
Only uses the standard library, so it runs without installing anything.
"""
import argparse
import csv
import datetime as dt
import json
import time
import urllib.request
from pathlib import Path

API_URL = "https://api.open-meteo.com/v1/forecast"
LATITUDE = 46.20
LONGITUDE = 6.14
VARIABLES = ["temperature_2m", "sunshine_duration", "wind_speed_10m"]
USER_AGENT = "lake-geneva-temp (student project, github.com/mariamicioni/lake-geneva-temp)"


def fetch() -> dict:
    url = (
        f"{API_URL}?latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&hourly={','.join(VARIABLES)}&forecast_days=3&timezone=Europe/Zurich"
    )
    # Try a few times in case the API is briefly unavailable.
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)["hourly"]
        except (OSError, ValueError, KeyError) as error:
            if attempt == 3:
                raise
            print(f"Attempt {attempt} failed ({error}), retrying in 60 s")
            time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw", help="output folder")
    args = parser.parse_args()

    now = dt.datetime.now(dt.timezone.utc)
    hourly = fetch()

    path = Path(args.out) / f"weather_geneva_fetched_{now.date()}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fetched_at_utc", "time_local", *VARIABLES])
        for i, time_local in enumerate(hourly["time"]):
            writer.writerow([now.isoformat(timespec="seconds"), time_local, *(hourly[v][i] for v in VARIABLES)])
    print(f"Saved {len(hourly['time'])} forecast hours to {path}")


if __name__ == "__main__":
    main()
