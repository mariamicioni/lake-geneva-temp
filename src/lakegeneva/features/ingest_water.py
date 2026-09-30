"""Download raw water readings of BAFU station 2606 (Lake Geneva outflow, Geneva).

Each run writes one CSV snapshot of the last N days. Snapshots overlap on purpose,
so a missed run loses nothing; duplicates are removed later in the feature pipeline.
Only uses the standard library, so it runs without installing anything.
"""
import argparse
import csv
import datetime as dt
import json
import time
import urllib.request
from pathlib import Path

API_URL = "https://api.existenz.ch/apiv1/hydro/daterange"
STATION = "2606"
PARAMETERS = "temperature,flow"
USER_AGENT = "lake-geneva-temp (student project, github.com/mariamicioni/lake-geneva-temp)"


def fetch(start: dt.date, end: dt.date) -> dict[int, dict[str, float]]:
    """Return {unix timestamp: {"temperature": ..., "flow": ...}} for the date range."""
    url = (
        f"{API_URL}?locations={STATION}&parameters={PARAMETERS}"
        f"&startdate={start}&enddate={end}&app=lake-geneva-temp"
    )
    # The API sometimes answers with an empty body for a short time, so try a few times.
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)["payload"]
            break
        except (OSError, ValueError, KeyError) as error:
            if attempt == 3:
                raise
            print(f"Attempt {attempt} failed ({error}), retrying in 60 s")
            time.sleep(60)

    readings: dict[int, dict[str, float]] = {}
    for item in payload:
        readings.setdefault(item["timestamp"], {})[item["par"]] = item["val"]
    return readings


def write_csv(readings: dict[int, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_utc", "water_temp_c", "flow_m3s"])
        for ts in sorted(readings):
            time_utc = dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat()
            writer.writerow([time_utc, readings[ts].get("temperature"), readings[ts].get("flow")])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="how many past days to download")
    parser.add_argument("--out", default="data/raw", help="output folder")
    args = parser.parse_args()

    today = dt.datetime.now(dt.timezone.utc).date()
    readings = fetch(today - dt.timedelta(days=args.days), today)
    if not readings:
        raise SystemExit("The API returned no readings.")

    path = Path(args.out) / f"water_{STATION}_fetched_{today}.csv"
    write_csv(readings, path)
    print(f"Saved {len(readings)} readings to {path}")


if __name__ == "__main__":
    main()
