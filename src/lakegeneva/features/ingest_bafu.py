"""Download daily mean water temperature and discharge of BAFU station 2606.

Source: the official BAFU Data Platform (GraphQL, no key), history back to 2010.
The same script is used for the daily run and for the backfill, so both use the
same source and the same definition of a daily mean:

    python src/lakegeneva/features/ingest_bafu.py                       # last 7 days
    python src/lakegeneva/features/ingest_bafu.py --start 2016-01-01    # backfill

Uses only the standard library and one small helper of this project (no extra packages).
"""
# Why one script for both jobs: the course calls this "backfilling" (Feature Store
# slides, p.7-8): run the same feature pipeline with a start and end date over old
# data. If live data and history came from two different scripts, they could
# compute the daily mean differently and the model would learn from data that
# does not look like what it gets in production (training-serving skew).

# All imports except the last one are from the Python standard library.
import argparse  # reads the command-line options --start, --end, --out
import csv  # writes the result as a CSV file
import datetime as dt  # dates, times and time zones
import json  # builds the request body and reads the API answer
import time  # time.sleep() to wait before a retry
import urllib.request  # sends the HTTP request to the API
from pathlib import Path  # file paths that work on Windows and Linux

from lakegeneva.common import raw_files  # a second run never replaces an earlier file

# --- Settings -----------------------------------------------------------------
# GraphQL APIs have one single URL; what we want is described in the query text.
API_URL = "https://data.bafu.admin.ch/api"
# Station 2606 = Rhône, Genève Halle de l'Île: where Lake Geneva flows out.
STATION = "2606"
# BAFU parameter codes -> the column names we use in our CSV.
# WT = water temperature (°C), Q = discharge / flow (m³/s).
PARAMETERS = {"WT": "water_temp_c", "Q": "flow_m3s"}
# Polite identification: tells the API owner who is calling and how to reach us.
USER_AGENT = "lake-geneva-temp (student project, github.com/mariamicioni/lake-geneva-temp)"

# BAFU stamps a daily mean with the start of the day in Swiss winter time (UTC+1),
# all year round: 2026-10-09T23:00:00Z is the mean of 10.10.2026.
# So we need a fixed UTC+1 time zone (not "Europe/Zurich", which switches to
# UTC+2 in summer and would shift every summer day by one).
BAFU_TIME = dt.timezone(dt.timedelta(hours=1))


def build_query(start: dt.date, end: dt.date) -> str:
    """GraphQL query for all daily means from start to end (both included)."""
    # The API filters on UTC timestamps, so we turn our Swiss days into UTC:
    # - first: midnight (UTC+1) at the start of the first day,
    #   e.g. 01.10.2026 00:00 UTC+1 = 2026-09-30T23:00:00Z
    first = dt.datetime.combine(start, dt.time(), BAFU_TIME).astimezone(dt.timezone.utc)
    # - after_last: midnight at the start of the day AFTER the last day.
    #   We ask for timestamps "< after_last", so the last day is included
    #   but the next one is not.
    after_last = dt.datetime.combine(end + dt.timedelta(days=1), dt.time(), BAFU_TIME).astimezone(dt.timezone.utc)
    # The query in words: "from the water observations, give me the daily means
    # (data_1day_mean) where the station number is 2606, the parameter is WT or Q,
    # and the timestamp is >= first and < after_last; for each row return the
    # timestamp, the parameter name and the value."
    # Note: in an f-string, {{ and }} produce literal { and } characters.
    return (
        "{ water { observations { data_1day_mean(where: {"
        f'station: {{no: {{_eq: "{STATION}"}}}}, '
        f"parameterName: {{_in: {json.dumps(list(PARAMETERS))}}}, "
        f'timestamp: {{_gte: "{first:%Y-%m-%dT%H:%M:%SZ}", _lt: "{after_last:%Y-%m-%dT%H:%M:%SZ}"}}'
        "}) { timestamp parameterName value } } } }"
    )


def fetch(start: dt.date, end: dt.date) -> dict[dt.date, dict[str, float]]:
    """Return {day: {"water_temp_c": ..., "flow_m3s": ...}} for the date range."""
    # GraphQL requests are a POST with a JSON body {"query": "..."}; .encode()
    # turns the text into bytes, which is what urllib sends.
    body = json.dumps({"query": build_query(start, end)}).encode()
    # Try a few times in case the API is briefly unavailable.
    for attempt in range(1, 4):  # attempts 1, 2 and 3
        try:
            request = urllib.request.Request(
                API_URL, data=body, headers={"Content-Type": "application/json", "User-Agent": USER_AGENT}
            )
            # timeout=120: a 10-year backfill is one big answer, so allow 2 minutes.
            with urllib.request.urlopen(request, timeout=120) as response:
                answer = json.load(response)
            # GraphQL reports problems inside the answer (with HTTP status 200),
            # so we have to check the "errors" field ourselves.
            if answer.get("errors"):
                raise ValueError(answer["errors"][0]["message"])
            # The rows sit deep inside the answer, mirroring the query structure.
            rows = answer["data"]["water"]["observations"]["data_1day_mean"]
            break  # success: leave the retry loop
        except (OSError, ValueError, KeyError, TypeError) as error:
            # OSError: network problem or timeout; ValueError: invalid JSON or a
            # GraphQL error; KeyError/TypeError: the answer has an unexpected shape.
            if attempt == 3:
                raise  # third failure: give up and let the job fail visibly
            print(f"Attempt {attempt} failed ({error}), retrying in 60 s")
            time.sleep(60)

    # The API sends one row per (day, parameter), in no particular order, e.g.
    #   {"timestamp": "2026-10-09T23:00:00Z", "parameterName": "WT", "value": 19.04}
    # We regroup them into one entry per day with both values:
    #   {date(2026, 10, 10): {"water_temp_c": 19.04, "flow_m3s": 124.946}}
    days: dict[dt.date, dict[str, float]] = {}
    for row in rows:
        # UTC timestamp -> UTC+1 -> calendar day (23:00Z on the 9th = the 10th).
        day = dt.datetime.fromisoformat(row["timestamp"]).astimezone(BAFU_TIME).date()
        # setdefault creates an empty dict for a day we have not seen yet.
        days.setdefault(day, {})[PARAMETERS[row["parameterName"]]] = row["value"]
    return days


def write_csv(days: dict[dt.date, dict[str, float]], path: Path) -> None:
    # Create the output folder if it does not exist yet (no error if it does).
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" stops Windows from writing an empty line after every row.
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", *PARAMETERS.values()])  # header: date,water_temp_c,flow_m3s
        for day in sorted(days):  # oldest day first
            # .get() returns None for a missing value; csv writes None as an
            # empty cell, so a missing measurement stays visibly empty.
            writer.writerow([day.isoformat(), *(days[day].get(column) for column in PARAMETERS.values())])


def main() -> None:
    # Command-line options; __doc__ (the text at the top of this file) becomes
    # the help text shown by --help.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # type=dt.date.fromisoformat turns "2016-01-01" into a date object.
    parser.add_argument("--start", type=dt.date.fromisoformat, help="first day, YYYY-MM-DD (default: 7 days before --end)")
    parser.add_argument("--end", type=dt.date.fromisoformat, help="last day, YYYY-MM-DD (default: today)")
    parser.add_argument("--out", default="data/raw", help="output folder")
    args = parser.parse_args()

    # Defaults: "today" in BAFU's time (UTC+1), and the 7 days before it.
    # Downloading 7 days every day overlaps on purpose: if a daily run is
    # missed, the next run still fetches the missing days.
    end = args.end or dt.datetime.now(BAFU_TIME).date()
    start = args.start or end - dt.timedelta(days=7)
    days = fetch(start, end)
    if not days:
        # SystemExit stops the script with an error, so the GitHub job turns red.
        raise SystemExit("The API returned no daily values.")

    # The file name records the requested range, e.g.
    # bafu_2606_daily_2016-01-01_to_2026-10-11.csv
    # A second run with the same range gets its own file (see common/raw_files.py).
    path = raw_files.new_path(Path(args.out) / f"bafu_{STATION}_daily_{start}_to_{end}.csv")
    write_csv(days, path)
    print(f"Saved {len(days)} days ({min(days)} to {max(days)}) to {path}")


# Runs main() only when the file is started as a script, not when it is imported
# (the tests import this file to test single functions).
if __name__ == "__main__":
    main()
