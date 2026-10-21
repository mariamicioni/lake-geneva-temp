"""Download raw water readings of BAFU station 2606 (Lake Geneva outflow, Geneva).

Each run writes one CSV snapshot of the last N days. Snapshots overlap on purpose,
so a missed run loses nothing; duplicates are removed later in the feature pipeline.
Only uses the standard library, so it runs without installing anything.
"""
# What this file is for: the 10-minute readings (144 per day). The proposal's
# features need them: "mean water temperature today until 18:00" and "last reading
# before 18:00" cannot be computed from BAFU's full-day means (see ingest_bafu.py).
# existenz.ch republishes the BAFU readings as a simple REST API, but only keeps
# about 32 days, which is why the history comes from ingest_bafu.py instead.
# Fallback: existenz.ch is run privately with no uptime guarantee (it failed on
# 19.10 and 21.10.2026). If it does not answer, the same 10-minute values are
# taken from the official BAFU Data Platform, so the daily file is still written.

# All imports are from the Python standard library (nothing to install).
import argparse  # reads the command-line options --days and --out
import csv  # writes the result as a CSV file
import datetime as dt  # dates, times and time zones
import json  # reads the API answer
import time  # time.sleep() to wait before a retry
import urllib.request  # sends the HTTP request to the API
from pathlib import Path  # file paths that work on Windows and Linux

# --- Settings -----------------------------------------------------------------
# "daterange" endpoint: all readings between a start and an end date.
API_URL = "https://api.existenz.ch/apiv1/hydro/daterange"
# Station 2606 = Rhône, Genève Halle de l'Île: where Lake Geneva flows out.
STATION = "2606"
# existenz.ch parameter names: water temperature (°C) and flow / discharge (m³/s).
PARAMETERS = "temperature,flow"
# Polite identification: tells the API owner who is calling.
USER_AGENT = "lake-geneva-temp (student project, github.com/mariamicioni/lake-geneva-temp)"
# Fallback source: the official BAFU Data Platform (GraphQL, no key, same API as
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
    """Same as fetch(), but from the official BAFU Data Platform (the fallback)."""
    # data_10min_mean = one value per 10 minutes, the same times as existenz.ch.
    # Checked on 21.10.2026: the temperature is identical; the flow differs by a
    # few m³/s, because BAFU averages the 10 minutes and existenz.ch gives the
    # single reading. Fine for a backup, the daily mean is almost the same.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7, help="how many past days to download")
    parser.add_argument("--out", default="data/raw", help="output folder")
    args = parser.parse_args()

    today = dt.datetime.now(dt.timezone.utc).date()
    # The API treats enddate as 00:00 of that day, so ask until tomorrow to include today.
    start, end = today - dt.timedelta(days=args.days), today + dt.timedelta(days=1)
    try:
        readings = fetch(start, end)
    except (OSError, ValueError, KeyError) as error:
        # existenz.ch failed 3 times: take the same readings from BAFU instead.
        # The message shows up in the GitHub log, so we know which source was used.
        print(f"existenz.ch failed ({error}), using the BAFU Data Platform instead")
        readings = fetch_bafu(start, end)
    if not readings:
        # SystemExit stops the script with an error, so the GitHub job turns red.
        raise SystemExit("The API returned no readings.")

    # One file per run, named after the download day, e.g. water_2606_fetched_2026-10-20.csv
    path = Path(args.out) / f"water_{STATION}_fetched_{today}.csv"
    write_csv(readings, path)
    print(f"Saved {len(readings)} readings to {path}")


# Runs main() only when the file is started as a script, not when it is imported
# (the tests import this file to test single functions).
if __name__ == "__main__":
    main()
