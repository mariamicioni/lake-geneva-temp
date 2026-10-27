"""One shared definition of the daily features and the label.

The forecast is made every day at 18:00 Swiss time (the "issue day" D) for the mean
water temperature of the next day (D+1). Training and the live forecast both build
their table with this file, so a feature can never be computed in two ways.

So far: the water part, from the 10-minute readings of station 2606.

    python src/lakegeneva/common/features.py    # builds the table from data/raw
"""
# Why one shared file: the Repository Guide asks for "one shared feature definition
# for training and inference" (FTI pipelines). If training and the live forecast
# had their own copies, a small difference (another time zone, another cutoff)
# would give the model data in production that does not look like its training
# data (training-serving skew).
# Why the 18:00 cutoff in the code: a feature may only use what was known at 18:00
# ("point-in-time correctness", Feature Store slides, p.9-10). The cutoff is applied
# to the time of each reading, so it does not matter when the job runs (GitHub
# started the daily job between 19:33 and 00:36 Swiss time in October 2026).

from pathlib import Path  # file paths that work on Windows and Linux

import pandas as pd  # tables: read the CSVs, group the readings by day

# --- Settings -----------------------------------------------------------------
# All days are Swiss calendar days. "Europe/Zurich" knows summer and winter time,
# so 18:00 is always 18:00 on the clock in Geneva.
TIMEZONE = "Europe/Zurich"
# The forecast is made at 18:00: only readings before 18:00 go into the features.
CUTOFF_HOUR = 18
# Label rule from the proposal: tomorrow counts only with at least 120 of its 144
# readings (83.3 %). Clock-change days have 138 or 150 readings and still pass.
MIN_READINGS_LABEL = 120
# The same 83.3 % for today before 18:00: at least 90 of the 108 readings
# (00:00 to 17:50). Otherwise no forecast is made from a half-empty day.
MIN_READINGS_D0 = 90
# The water features of the proposal, in the order of the table.
WATER_FEATURES = ["water_mean_d0", "water_last", "water_mean_3d", "water_trend_3d", "flow_mean_d0"]
# The column with the value we want to predict.
LABEL = "label_water_mean_d1"


def load_water(raw_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Read all 10-minute water files into one table in Swiss time, oldest first."""
    # The backfill and every daily run wrote their own CSV with the same columns,
    # and they overlap on purpose. sorted() reads them in name order: the backfill
    # (water_2606_10min_...) first, then the daily files (water_2606_fetched_<date>)
    # from old to new.
    paths = sorted(Path(raw_dir).glob("water_2606_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No water files (water_2606_*.csv) in {raw_dir}")
    # Read every file and stack them into one table.
    readings = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    # Text -> real time stamps. utc=True: the times in the raw files are UTC.
    readings["time_utc"] = pd.to_datetime(readings["time_utc"], utc=True, format="ISO8601")
    # The same reading can be in several files. keep="last" keeps the copy from the
    # file that comes last in name order, i.e. the newest daily download. The two
    # sources differ by less than 0.01 °C on average (checked on 27.10.2026).
    readings = readings.drop_duplicates(subset="time_utc", keep="last")
    # Swiss local time (with summer time): this is the clock of the 18:00 cutoff.
    readings["time_local"] = readings["time_utc"].dt.tz_convert(TIMEZONE)
    readings = readings.sort_values("time_local").reset_index(drop=True)
    return readings[["time_local", "water_temp_c", "flow_m3s"]]


def water_features(readings: pd.DataFrame) -> pd.DataFrame:
    """Daily table: one row per Swiss day D with the water features at 18:00 and the label."""
    # Oldest reading first, so that "last" below really is the latest reading.
    readings = readings.sort_values("time_local")
    local = readings["time_local"]
    # The Swiss calendar day of each reading, as a plain date: tz_localize(None)
    # drops the time zone but keeps the Swiss clock time, normalize() sets it to 00:00.
    day = local.dt.tz_localize(None).dt.normalize()
    # True for the readings that were already measured at 18:00 of their day.
    before_cutoff = local.dt.hour < CUTOFF_HOUR

    # Per day: how many temperature readings there are and their sum ...
    # ... over the whole day (for the label and for the previous days of the 3-day mean),
    whole_day = readings["water_temp_c"].groupby(day).agg(["count", "sum"])
    # ... and only before 18:00 (for today's features).
    early = readings[before_cutoff].groupby(day[before_cutoff]).agg(
        n_d0=("water_temp_c", "count"),
        sum_d0=("water_temp_c", "sum"),
        water_last=("water_temp_c", "last"),  # the latest reading before 18:00
        flow_mean_d0=("flow_m3s", "mean"),
    )
    # One row for every calendar day, also for days without any reading, so that
    # shift(1) below always means "one day earlier".
    days = pd.date_range(day.min(), day.max(), freq="D", name="date")
    whole_day = whole_day.reindex(days)
    early = early.reindex(days)

    table = pd.DataFrame(index=days)
    # How many readings of today exist before 18:00 (108 on a normal day).
    table["n_readings_d0"] = early["n_d0"].fillna(0).astype(int)
    # Days that pass the completeness rule; all other days get empty features.
    complete = table["n_readings_d0"] >= MIN_READINGS_D0

    # --- Features: only what is known at 18:00 of day D ----------------------------
    # water_mean_d0: mean water temperature today from 00:00 to 18:00.
    table["water_mean_d0"] = (early["sum_d0"] / early["n_d0"]).where(complete)
    # water_last: the last reading before 18:00.
    table["water_last"] = early["water_last"].where(complete)
    # water_mean_3d: mean of all readings of the day before yesterday and of
    # yesterday (both whole days, already in the past) and of today until 18:00.
    sum_3d = whole_day["sum"].shift(2) + whole_day["sum"].shift(1) + early["sum_d0"]
    count_3d = whole_day["count"].shift(2) + whole_day["count"].shift(1) + early["n_d0"]
    table["water_mean_3d"] = (sum_3d / count_3d).where(complete)
    # water_trend_3d: today's 00:00-18:00 mean minus the same hours three days
    # earlier. The same hours on both days, so the warm-up in the afternoon does
    # not look like a trend.
    table["water_trend_3d"] = table["water_mean_d0"] - table["water_mean_d0"].shift(3)
    # flow_mean_d0: mean discharge (m³/s) today until 18:00.
    table["flow_mean_d0"] = early["flow_mean_d0"].where(complete)

    # --- Label: what we want to predict ---------------------------------------------
    # The mean of all readings of TOMORROW (D+1). shift(-1) moves tomorrow's values
    # into today's row. It is a measurement from the future, so it is never used as
    # a feature, and it is empty while tomorrow is not over yet.
    table["n_readings_d1"] = whole_day["count"].shift(-1).fillna(0).astype(int)
    tomorrow_mean = (whole_day["sum"] / whole_day["count"]).shift(-1)
    table[LABEL] = tomorrow_mean.where(table["n_readings_d1"] >= MIN_READINGS_LABEL)
    return table


def main() -> None:
    # Build the table from all raw water files and show the last days. The next
    # step saves it as Parquet (the feature store) instead of printing it.
    table = water_features(load_water("data/raw"))
    print(f"{len(table)} days, {table.index.min():%d.%m.%Y} to {table.index.max():%d.%m.%Y}")
    print(table.tail(5).round(3).to_string())


# Runs main() only when the file is started as a script, not when it is imported
# (the tests and, later, the training pipeline import this file).
if __name__ == "__main__":
    main()
