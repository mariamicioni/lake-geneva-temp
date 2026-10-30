"""Download raw water readings of BAFU station 2606 (Lake Geneva outflow, Geneva).

Daily run: one CSV snapshot of the last N days, from the BAFU Data Platform (with
existenz.ch as fallback). Snapshots overlap on purpose, so a missed run loses
nothing; duplicates are removed later in the feature pipeline.
Backfill: the same 10-minute readings for any date range (--start/--end), from the
BAFU Data Platform, because existenz.ch only keeps about 32 days:

    python src/lakegeneva/features/ingest_water.py                       # last 7 days
    python src/lakegeneva/features/ingest_water.py --start 2016-01-01    # backfill

Only uses the standard library, so it runs without installing anything.
"""
# What this file is for: the 10-minute readings (144 per day). The proposal's
# features need them: "mean water temperature today until 18:00" and "last reading
# before 18:00" cannot be computed from BAFU's full-day means (see ingest_bafu.py).
# Main source (since 27.10.2026): the official BAFU Data Platform, the same source
# as the 10-minute history (see "Backfill" below), so the live data and the
# training data come from the same place. Its newest value is about 1.5 hours old
# (measured on 27.10.2026), so the daily job runs well after the 18:00 cutoff.
# Fallback: existenz.ch republishes the same BAFU readings as a simple REST API
# (single readings instead of 10-minute means: on average 0.005-0.008 °C apart,
# checked on 27.10.2026). It keeps only about 32 days and is run privately with no
# uptime guarantee (it failed on 19.10 and 21.10.2026), so it is only the backup.
# ingest_bafu.py adds the BAFU daily means back to 2016.
# Backfill: the 18:00 features and the label rule (at least 120 of 144 readings)
# need 10-minute readings for every training day, not only daily means. So the
# history comes through this same script and is written in the same CSV format as
# the daily files - the course calls this "backfilling" (Feature Store slides,
# p.7-8): one pipeline, run with a start and end date over old data.

# All imports are from the Python standard library (nothing to install).
import argparse  # reads the command-line options --days, --start, --end and --out
import csv  # writes the result as a CSV file
import datetime as dt  # dates, times and time zones
import json  # reads the API answer
import time  # time.sleep() to wait before a retry
import urllib.request  # sends the HTTP request to the API
from pathlib import Path  # file paths that work on Windows and Linux

# --- Settings -----------------------------------------------------------------
# Fallback source, existenz.ch: the "daterange" endpoint gives all readings
# between a start and an end date.
API_URL = "https://api.existenz.ch/apiv1/hydro/daterange"
# Station 2606 = Rhône, Genève Halle de l'Île: where Lake Geneva flows out.
STATION = "2606"
# existenz.ch parameter names: water temperature (°C) and flow / discharge (m³/s).
PARAMETERS = "temperature,flow"
# Polite identification: tells the API owner who is calling.
USER_AGENT = "lake-geneva-temp (student project, github.com/mariamicioni/lake-geneva-temp)"
# Main source: the official BAFU Data Platform (GraphQL, no key, same API as
# ingest_bafu.py). BAFU parameter codes -> the names existenz.ch uses, so both
# sources give the same dict and write_csv() does not need to know the source.
BAFU_API_URL = "https://data.bafu.admin.ch/api"
BAFU_PARAMETERS = {"WT": "temperature", "Q": "flow"}


def fetch(start: dt.date, end: dt.date) -> dict[int, dict[str, float]]:
    """Return {unix timestamp: {"temperature": ..., "flow": ...}} for the date range."""
    # REST API: everything we want is in the URL (query parameters after "?").
    # app=lake-geneva-temp is asked for by existenz.ch to identify the app.
    url = (
        f"{API_URL}?locations={STATION}&parameters={PARAMETERS}"
        f"&startdate={start}&enddate={end}&app=lake-geneva-temp"
    )
    # The API sometimes answers with an empty body for a short time, so try a few times.
    for attempt in range(1, 4):  # attempts 1, 2 and 3
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                # The readings are in the "payload" list of the JSON answer.
                payload = json.load(response)["payload"]
            break  # success: leave the retry loop
        except (OSError, ValueError, KeyError) as error:
            # OSError: network problem or timeout; ValueError: empty or invalid
            # JSON; KeyError: the answer has no "payload".
            if attempt == 3:
                raise  # third failure: give up and let the job fail visibly
            print(f"Attempt {attempt} failed ({error}), retrying in 60 s")
            time.sleep(60)

    # The API sends one item per (time, parameter), e.g.
    #   {"timestamp": 1760000000, "par": "temperature", "val": 18.5}
    # We regroup them into one entry per time with both values:
    #   {1760000000: {"temperature": 18.5, "flow": 250.0}}
    # The timestamp is a Unix time: seconds since 01.01.1970 00:00 UTC.
    readings: dict[int, dict[str, float]] = {}
    for item in payload:
        # setdefault creates an empty dict for a timestamp we have not seen yet.
        readings.setdefault(item["timestamp"], {})[item["par"]] = item["val"]
    return readings


def fetch_bafu(start: dt.date, end: dt.date) -> dict[int, dict[str, float]]:
    """Same as fetch(), but from the official BAFU Data Platform (the main source)."""
    # data_10min_mean = one value per 10 minutes, the same times as existenz.ch.
    # BAFU averages the 10 minutes, existenz.ch gives the single reading: the
    # temperature differs by 0.005-0.008 °C on average, the flow by a few m³/s
    # (checked on 21.10 and 27.10.2026).
    # The time window is from 00:00 UTC of start to 00:00 UTC of end (end excluded),
    # like the existenz.ch request. Note: {{ and }} produce literal { and } in an f-string.
    query = (
        "{ water { observations { data_10min_mean(where: {"
        f'station: {{no: {{_eq: "{STATION}"}}}}, '
        f"parameterName: {{_in: {json.dumps(list(BAFU_PARAMETERS))}}}, "
        f'timestamp: {{_gte: "{start}T00:00:00Z", _lt: "{end}T00:00:00Z"}}'
        "}) { timestamp parameterName value } } } }"
    )
    # GraphQL requests are a POST with a JSON body {"query": "..."}.
    body = json.dumps({"query": query}).encode()
    for attempt in range(1, 4):  # attempts 1, 2 and 3
        try:
            request = urllib.request.Request(
                BAFU_API_URL, data=body, headers={"Content-Type": "application/json", "User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                answer = json.load(response)
            # GraphQL reports problems inside the answer (with HTTP status 200).
            if answer.get("errors"):
                raise ValueError(answer["errors"][0]["message"])
            rows = answer["data"]["water"]["observations"]["data_10min_mean"]
            break  # success: leave the retry loop
        except (OSError, ValueError, KeyError, TypeError) as error:
            if attempt == 3:
                raise  # third failure: give up and let the job fail visibly
            print(f"BAFU attempt {attempt} failed ({error}), retrying in 60 s")
            time.sleep(60)

    # One row per (time, parameter), e.g.
    #   {"timestamp": "2026-09-30T10:00:00Z", "parameterName": "WT", "value": 20.63}
    # regrouped like fetch() does, with the time as a Unix timestamp:
    #   {1759226400: {"temperature": 20.63, "flow": 102.779}}
    readings: dict[int, dict[str, float]] = {}
    for row in rows:
        ts = int(dt.datetime.fromisoformat(row["timestamp"]).timestamp())
        readings.setdefault(ts, {})[BAFU_PARAMETERS[row["parameterName"]]] = row["value"]
    return readings


def fetch_bafu_history(start: dt.date, end: dt.date, chunk_days: int = 30) -> dict[int, dict[str, float]]:
    """Same as fetch_bafu(), for a long range: downloads it in pieces of 30 days."""
    # Ten years of 10-minute values are over a million rows, too much for one
    # request. A piece of 30 days is 30 x 144 readings x 2 parameters = 8,640 rows
    # (tested on 27.10.2026: about 4 seconds per piece).
    readings: dict[int, dict[str, float]] = {}
    piece_start = start
    while piece_start < end:
        # The last piece can be shorter, so that it never goes past `end`.
        piece_end = min(piece_start + dt.timedelta(days=chunk_days), end)
        # The pieces do not overlap (each one stops where the next one starts), so
        # update() simply adds the new readings to the ones we already have.
        readings.update(fetch_bafu(piece_start, piece_end))
        print(f"Downloaded {piece_start} to {piece_end}: {len(readings)} readings so far")
        piece_start = piece_end
        time.sleep(1)  # a short pause between requests, to be polite to the API
    return readings


def write_csv(readings: dict[int, dict[str, float]], path: Path) -> None:
    # Create the output folder if it does not exist yet (no error if it does).
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" stops Windows from writing an empty line after every row.
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_utc", "water_temp_c", "flow_m3s"])
        for ts in sorted(readings):  # oldest reading first
            # Unix time -> readable UTC time, e.g. 2025-10-09T08:53:20+00:00.
            # We keep UTC on purpose: one clear time zone in the raw data; the
            # conversion to Swiss time (18:00 cutoff) happens in the features.
            time_utc = dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat()
            # .get() returns None for a missing value, which csv writes as an
            # empty cell, so a missing measurement stays visibly empty.
            writer.writerow([time_utc, readings[ts].get("temperature"), readings[ts].get("flow")])


def main() -> None:
    # Command-line options; __doc__ (the text at the top) becomes the --help text.
    # RawDescriptionHelpFormatter keeps the line breaks of the examples in --help.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=7, help="daily run: how many past days to download")
    # type=dt.date.fromisoformat turns "2016-01-01" into a date object.
    parser.add_argument("--start", type=dt.date.fromisoformat, help="backfill: first day, YYYY-MM-DD")
    parser.add_argument("--end", type=dt.date.fromisoformat, help="backfill: last day, YYYY-MM-DD (default: today)")
    parser.add_argument("--out", default="data/raw", help="output folder")
    args = parser.parse_args()
    if args.end and not args.start:
        # parser.error() prints the message and stops, like a wrong option does.
        parser.error("--end only works together with --start")

    today = dt.datetime.now(dt.timezone.utc).date()
    if args.start:
        # Backfill: only the BAFU Data Platform has the history.
        end = args.end or today
        # Our days are Swiss days, but the API works in UTC. A Swiss day starts at
        # 22:00 or 23:00 UTC of the day before, so we start one UTC day earlier and
        # stop at the end of the UTC day `end`: every Swiss hour is included. The
        # extra hours do no harm, the features keep only what they need.
        readings = fetch_bafu_history(args.start - dt.timedelta(days=1), end + dt.timedelta(days=1))
        # One file for the whole range, e.g. water_2606_10min_2016-01-01_to_2026-10-27.csv
        path = Path(args.out) / f"water_{STATION}_10min_{args.start}_to_{end}.csv"
    else:
        # Daily run, BAFU first (the same source as the training history, so no
        # training-serving skew). Both APIs treat the end as 00:00 of that day, so
        # ask until tomorrow to include today.
        start, end = today - dt.timedelta(days=args.days), today + dt.timedelta(days=1)
        try:
            readings = fetch_bafu(start, end)
            # An empty answer is not an error for the API, but it is useless for us:
            # raising here sends it to the same fallback as a failed request.
            if not readings:
                raise ValueError("no readings in the answer")
        except (OSError, ValueError, KeyError, TypeError) as error:
            # BAFU failed 3 times or sent nothing: take the same readings from existenz.ch.
            # The message shows up in the GitHub log, so we know which source was used.
            print(f"BAFU Data Platform failed ({error}), using existenz.ch instead")
            readings = fetch(start, end)
        # One file per run, named after the download day, e.g. water_2606_fetched_2026-10-20.csv
        path = Path(args.out) / f"water_{STATION}_fetched_{today}.csv"
    if not readings:
        # SystemExit stops the script with an error, so the GitHub job turns red.
        raise SystemExit("The API returned no readings.")

    write_csv(readings, path)
    print(f"Saved {len(readings)} readings to {path}")


# Runs main() only when the file is started as a script, not when it is imported
# (the tests import this file to test single functions).
if __name__ == "__main__":
    main()
