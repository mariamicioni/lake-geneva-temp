"""One shared definition of the daily features and the label.

Every evening (the "issue day" D) the forecast for the mean water temperature of the
next day (D+1) uses only the data from before 18:00 Swiss time. Training and the live
forecast both build their table with this file, so a feature can never be computed
in two ways.

Water part: from the 10-minute readings of station 2606 (ingest_water.py).
Weather part: from the 48-hour-old Open-Meteo forecast (ingest_weather_runs.py).

    python src/lakegeneva/common/features.py    # builds the table from data/raw
"""
# Why one shared file: the Repository Guide asks for "one shared feature definition
# for training and inference" (FTI pipelines). If training and the live forecast
# had their own copies, a small difference (another time zone, another cutoff)
# would give the model data in production that does not look like its training
# data (training-serving skew).
# Why the 18:00 cutoff in the code: a feature may only use data from before the
# cutoff ("point-in-time correctness", Feature Store slides, p.9-10). The cutoff is
# applied to the time of each reading, so it does not matter when the job runs (GitHub
# started the daily job between 19:33 and 00:36 Swiss time in October 2026).
# 18:00 is the cutoff for the data, not the time the forecast is ready: BAFU publishes
# its readings about 1.5 hours late, so the readings from just before 18:00 only
# exist online after about 19:30. The forecast therefore comes after the evening job,
# and training and the live forecast use the same readings.

from pathlib import Path  # file paths that work on Windows and Linux

import numpy as np  # cosine for the south-west wind component
import pandas as pd  # tables: read the CSVs, group the readings by day

# --- Settings -----------------------------------------------------------------
# All days are Swiss calendar days. "Europe/Zurich" knows summer and winter time,
# so 18:00 is always 18:00 on the clock in Geneva.
TIMEZONE = "Europe/Zurich"
# The data cutoff: only readings measured before 18:00 go into the features.
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

# The four columns of the 48-hour-old forecast, as ingest_weather_runs.py saves them.
TEMPERATURE = "temperature_2m_previous_day2"  # air temperature 2 m above ground (°C)
SUNSHINE = "sunshine_duration_previous_day2"  # seconds of sunshine in the hour (0-3600)
WIND_SPEED = "wind_speed_10m_previous_day2"  # wind speed 10 m above ground (km/h)
WIND_DIRECTION = "wind_direction_10m_previous_day2"  # degrees the wind comes FROM
# Wind from the south-west (225°) pushes the warm surface water towards the
# north-east end of the lake, and cold deep water comes up at Geneva (upwelling,
# Eawag; review point A3). Checked on 27.10.2026: from April to September the
# south-west part of the wind goes with tomorrow's temperature change about twice
# as strongly as the wind speed alone (correlation -0.29 vs -0.13).
SOUTH_WEST = 225
# A forecast day counts only with at least 20 of its 24 hours (the same 83 % as
# the water rules). Clock-change days have 23 or 25 hours and still pass.
MIN_HOURS_FC = 20
# The weather features: tomorrow's forecast, plus today's south-west wind
# (upwelling needs some hours, so wind today can cool the water tomorrow).
WEATHER_FEATURES = ["air_temp_fc_d1", "sunshine_fc_d1", "wind_max_fc_d1", "wind_sw_fc_d0", "wind_sw_fc_d1"]
# All features of the model, in the order of the table.
FEATURES = WATER_FEATURES + WEATHER_FEATURES


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
    # The same reading can be in several files. For each time and each column, last()
    # takes the value from the file that comes last in name order, i.e. the newest
    # daily download, and it skips empty cells. So a newer file with a gap cannot delete
    # a good value of an older file (on 30.10.2026 this saved 5 temperature and 23 flow
    # readings). The two sources differ by less than 0.01 °C on average (checked on
    # 27.10.2026).
    readings = readings.groupby("time_utc", as_index=False)[["water_temp_c", "flow_m3s"]].last()
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

    # --- Features: only what was measured before 18:00 of day D --------------------
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
    # Tomorrow is over when the data reaches its last 10-minute slot (23:50 Swiss
    # time) or later. Counting the readings alone is not enough: the job of 27.10.2026
    # started at 23:52 and already had 126 readings of a day that was not over yet.
    # table.index + 2 days = 00:00 at the end of tomorrow; tz_localize puts it on the
    # Swiss clock (00:00 always exists, the clocks change at 02:00 or 03:00).
    end_of_tomorrow = (table.index + pd.Timedelta(days=2)).tz_localize(TIMEZONE)
    tomorrow_over = end_of_tomorrow - pd.Timedelta(minutes=10) <= local.max()
    table[LABEL] = tomorrow_mean.where((table["n_readings_d1"] >= MIN_READINGS_LABEL) & tomorrow_over)
    return table


def load_weather(raw_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Read all files of the 48-hour-old forecast into one table, one row per hour."""
    # Only the 48-hour-old forecasts (weather_geneva_prev2_...). The daily snapshots
    # of the newest forecast (weather_geneva_fetched_...) are an audit copy: they
    # were downloaded after 18:00 and must never become features.
    paths = sorted(Path(raw_dir).glob("weather_geneva_prev2_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No forecast files (weather_geneva_prev2_*.csv) in {raw_dir}")
    hours = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    hours["fetched_at_utc"] = pd.to_datetime(hours["fetched_at_utc"], utc=True)
    # Two file formats. Since 27.10.2026 the files hold UTC times (time_utc). The
    # first files hold time_local: for those, Open-Meteo shifted ALL hours by the UTC
    # offset that was valid at the moment of the download (+1 h in winter even for
    # summer days, checked on 27.10.2026). That shift is undone with the offset at
    # the download time, so every hour gets its real UTC time.
    if "time_utc" in hours:
        time_utc = pd.to_datetime(hours["time_utc"], utc=True)  # empty (NaT) in old files
    else:
        time_utc = pd.Series(pd.NaT, index=hours.index, dtype="datetime64[ns, UTC]")
    if "time_local" in hours:
        old = time_utc.isna() & hours["time_local"].notna()  # rows from the old files
        fetched = hours["fetched_at_utc"]
        # The offset in Switzerland at the download: Swiss clock time minus UTC (1 or 2 h).
        offset = fetched.dt.tz_convert(TIMEZONE).dt.tz_localize(None) - fetched.dt.tz_localize(None)
        shifted_back = (pd.to_datetime(hours["time_local"]) - offset).dt.tz_localize("UTC")
        time_utc = time_utc.where(~old, shifted_back)
    hours["time_utc"] = time_utc
    # Swiss local time WITH summer time: this decides to which Swiss day an hour belongs.
    hours["time_local"] = time_utc.dt.tz_convert(TIMEZONE)
    # The same hour can be in several files (the daily runs overlap). Keep the
    # newest download: sort by download time, then keep the last copy of each hour.
    # A 48-hour-old forecast does not change, so this only makes the choice clear.
    hours = hours.sort_values("fetched_at_utc").drop_duplicates(subset="time_utc", keep="last")
    return hours.sort_values("time_utc").reset_index(drop=True)


def weather_features(hours: pd.DataFrame) -> pd.DataFrame:
    """Daily table: one row per issue day D with the forecast for tomorrow (and today's wind)."""
    # The Swiss day each forecast hour belongs to, as a plain date (like in
    # water_features: drop the time zone but keep the Swiss clock time, then 00:00).
    day = hours["time_local"].dt.tz_localize(None).dt.normalize()
    # The south-west part of the wind in each hour (km/h). cos(0°) = 1: wind exactly
    # from the south-west counts fully; cos(180°) = -1: wind from the north-east
    # (the bise) counts negative; wind across the lake counts about 0. Each hour
    # becomes a number first and only these numbers are averaged: averaging the
    # angles themselves would be wrong (350° and 10° are both north, but their
    # average is 180° = south).
    angle = np.deg2rad(hours[WIND_DIRECTION] - SOUTH_WEST)
    south_west = hours[WIND_SPEED] * np.cos(angle)
    # An hour counts only if all four values exist (before 20.01.2024 sunshine and
    # wind are missing in the 48-hour-old forecast).
    complete_hour = hours[[TEMPERATURE, SUNSHINE, WIND_SPEED, WIND_DIRECTION]].notna().all(axis=1)

    # One row per day with the daily values of the forecast.
    per_hour = pd.DataFrame(
        {
            "temperature": hours[TEMPERATURE],
            "sunshine": hours[SUNSHINE],
            "wind_speed": hours[WIND_SPEED],
            "south_west": south_west,
            "complete": complete_hour,
        }
    )
    per_day = per_hour.groupby(day).agg(
        air_temp=("temperature", "mean"),  # mean air temperature of the day (°C)
        sunshine_s=("sunshine", "sum"),  # seconds of sunshine in the whole day
        wind_max=("wind_speed", "max"),  # strongest hourly wind of the day (km/h)
        wind_sw=("south_west", "mean"),  # mean south-west push of the day (km/h)
        n_hours=("complete", "sum"),  # hours with all four values
    )
    # One row for every calendar day, so that shift(-1) means "one day later".
    days = pd.date_range(per_day.index.min(), per_day.index.max(), freq="D", name="date")
    per_day = per_day.reindex(days)
    # A forecast day with fewer than 20 complete hours gives empty values
    # (where(..., axis=0) applies the yes/no of each day to all its columns).
    usable = per_day["n_hours"] >= MIN_HOURS_FC
    per_day = per_day[["air_temp", "sunshine_s", "wind_max", "wind_sw"]].where(usable, axis=0)

    table = pd.DataFrame(index=days)
    # Tomorrow's forecast: shift(-1) moves the values of day D+1 into the row of D.
    table["air_temp_fc_d1"] = per_day["air_temp"].shift(-1)
    table["sunshine_fc_d1"] = per_day["sunshine_s"].shift(-1) / 3600  # seconds -> hours
    table["wind_max_fc_d1"] = per_day["wind_max"].shift(-1)
    # Today's south-west wind uses all 24 hours of today, also those after 18:00:
    # these are forecasts made 48 hours earlier, not measurements, so they are
    # known at 18:00 (see ingest_weather_runs.py).
    table["wind_sw_fc_d0"] = per_day["wind_sw"]
    table["wind_sw_fc_d1"] = per_day["wind_sw"].shift(-1)
    return table[WEATHER_FEATURES]


def feature_table(raw_dir: Path | str = "data/raw") -> pd.DataFrame:
    """The full daily table: water features, weather features and the label."""
    water = water_features(load_water(raw_dir))
    weather = weather_features(load_weather(raw_dir))
    # join() matches the rows by date. how="left": every day of the water table
    # stays; a day without a forecast simply has empty weather features.
    table = water.join(weather, how="left")
    # Fixed column order: the counts, the features, the label.
    return table[["n_readings_d0", *FEATURES, "n_readings_d1", LABEL]]


def main() -> None:
    # Build the full table from all raw files and show the last days. The next
    # step saves it as Parquet (the feature store) instead of printing it.
    table = feature_table("data/raw")
    print(f"{len(table)} days, {table.index.min():%d.%m.%Y} to {table.index.max():%d.%m.%Y}")
    print(table.tail(5).round(2).to_string())


# Runs main() only when the file is started as a script, not when it is imported
# (the tests and, later, the training pipeline import this file).
if __name__ == "__main__":
    main()
