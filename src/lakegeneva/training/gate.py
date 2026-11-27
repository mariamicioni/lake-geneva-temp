"""The promotion gate: may a new model version become the champion?

Five checks, and the new version becomes the champion only if all of them pass:

- G1, validation: the training procedure beats the two free forecasts (plain
  persistence and persistence + the average change) on the validation days.
- G2, validation: the training procedure is not worse than the champion's procedure,
  on exactly the same validation days with the same labels.
- G3, shadow: on the newest days, the new model and the champion forecast almost the
  same temperature. No label is needed.
- G4, plausible: on the newest days, the new model never predicts a change bigger than
  the largest change before the test period.
- G5, fresh data: the row of the issue day D is ready for the newest evening (the
  same check as check_fresh.py) AND the newest label is recent, so the new version
  did not learn from a table that stopped growing.

No check reads a label of a test day (21.01.2025 onwards). The test days are only for
reporting the result (README, MLflow), never for a decision.
"""
# Why a new gate (review point B3, 15.11.2026; design: gate_design.md, checked by a
# second review on 17.11.2026). The old gate asked "is the whole 95 % interval of
# (model - persistence) on all test days below 0?". It made a decision from the test
# days, it could hardly fail (7 new days a week do not move an interval over 660 days),
# and it never looked at the champion.
#
# What this gate is for: it guards against BREAKAGE in a weekly job: a bug in the code,
# old data that changed, a broken feature, a wrong model. It does not prove that the
# new model is better on new days. That question needs days the model has never seen
# with their labels: the monitoring of the live forecast in MS4 answers it.
#
# What G1 and G2 really measure: the validation MAE comes from the walk-forward test on
# the validation days, with a new model for every day. So it measures the training
# PROCEDURE (code + old data), not the one model that is registered. alpha was also
# chosen on these days, so the number is a little too good. Both are fine for finding
# regressions, but it is not an independent proof of quality.
#
# Fail-closed: if a check cannot be done (MLflow gives no answer, a number is missing,
# a forecast is NaN), the answer is "no". Only a confirmed "there is no champion yet"
# lets G2 and G3 be skipped.

import datetime as dt  # the clock
import hashlib  # md5: the fingerprint of the validation days
import math  # math.isfinite: a number that is not NaN and not infinite
from zoneinfo import ZoneInfo  # the Swiss clock

import numpy as np  # np.isfinite on many numbers at once
import pandas as pd  # the feature table and the forecasts

from lakegeneva.common import baselines, features, forecasting  # shared definitions
from lakegeneva.features import check_fresh  # the issue-day check of the feature pipeline
from lakegeneva.training import experiments, train  # row 3 and the walk-forward test

# --- Settings -----------------------------------------------------------------
# All numbers were fixed from principles and from data before the test period, never
# from a test day (gate_design.md, 17.11.2026).
# G2: how much worse than the champion the validation MAE may be: about 1 % of the
# persistence MAE on the validation days (0.606 degC). With the same code and the same
# old data the validation MAE does not change at all, so any increase comes from a
# change in the code or in the old data.
MAX_MAE_INCREASE = 0.005
# G3: how far the new model's forecast may be from the champion's on one day. A normal
# weekly retrain moves a forecast by 0.015 degC on average and by at most 0.113 degC
# (validation days, 17.11.2026). 0.3 is almost 3 times that largest move.
MAX_SHADOW_DIFFERENCE = 0.3
# G4: the largest change from the last reading to tomorrow's mean in the 365 rows whose
# label was known at 18:00 on the first test day (21.01.2024-19.01.2025; it happened on
# 29.06.2024). It is fixed here as a number: taken from all training rows, it would come
# from a test-day label (8.29 degC on 06.07.2025), so a test day would set the limit
# (found in the review of 17.11.2026). A bigger predicted change is a sign of a broken
# model, not a proof.
MAX_CHANGE = 5.304583
# G3 and G4 look at the newest rows that have all features (with or without a label).
NEWEST_DAYS = 7
# G5: how old the newest row WITH A LABEL may be. Normal: on Monday morning (retrain.yml)
# it is Friday's row, 3 days old, because a label needs the whole next day. If the
# feature run of Sunday failed, it is Thursday's (4 days), while Saturday's row still
# passes the issue-day check (one day late). 5 gives one more day. Older means that the
# labels stopped coming although the features did not (review of 18.11.2026: with
# fresh features and no new label for 37 days, the gate still passed).
MAX_LABEL_AGE_DAYS = 5

# The five checks, in the order they run, with the names used in the MLflow tags.
CHECKS = (
    "g1_validation_vs_free_forecasts",
    "g2_validation_vs_champion",
    "g3_shadow_vs_champion",
    "g4_plausible_change",
    "g5_fresh_data",
)


def now() -> dt.datetime:
    """The time now, in UTC (a function, so the tests can set another time)."""
    return dt.datetime.now(dt.UTC)


def is_number(value: object) -> bool:
    """True for a real number that is not NaN and not infinite."""
    # bool is excluded on purpose: True would count as 1.
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def newest_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The newest NEWEST_DAYS rows that have all features, also if their label is not known yet."""
    # Why not train.model_rows(): it also needs the label, so it drops the newest two
    # days, exactly the days the live forecast is made for (found in the review of
    # 17.11.2026). G3 and G4 need no label: they compare the models' answers only.
    complete = table[features.FEATURES].notna().all(axis=1)
    return table[complete].sort_index().tail(NEWEST_DAYS)


def validation_summary(rows: pd.DataFrame, columns: list[str], alpha: float) -> dict:
    """The procedure, persistence and row 3 on the validation days, and a fingerprint of those days."""
    # The validation days: the validation year from baselines.VALIDATION_START up to the
    # rows whose label is known at 18:00 on the first test day (19.01.2025), exactly as
    # in train.choose_alpha (the same function). Only these labels are read. Before, it
    # was every complete row before the test days: the same today, but after a longer
    # backfill G1 would have judged other days than the validation year (found in a
    # review on 26.11.2026: on a test table it passed over 1,755 days and failed on 365).
    validation_days = baselines.validation_days(rows).index
    # The walk-forward test of the procedure: a new model for every validation day,
    # trained on the rows up to two days before it (the same as in choose_alpha).
    forecast = train.walk_forward(rows, columns, alpha, validation_days)
    part = rows.loc[forecast.index]  # the validation days that got a forecast
    truth = part[features.LABEL]
    # Row 3 of the README: the last reading plus the average change of the rows known
    # at 18:00. On a validation day these rows are all before the test period.
    row3 = experiments.average_change(rows, forecast.index)
    # The fingerprint of the validation days: the md5 of their dates, labels and last
    # readings as text. Two validation MAEs can only be compared if it is the same:
    # otherwise the days or the truth changed, not only the model (review of 17.11.2026).
    # The features are NOT in it on purpose: a change of the features (for example a bug
    # in features.py) must stay comparable, because that is what G2 is there to find.
    # lineterminator="\n": the same line ends on Windows and on Linux. pandas otherwise
    # uses the line end of the computer ("\r\n" on Windows), so the same days would get
    # two fingerprints, and a run on GitHub (Linux) could never be compared with a run
    # on a laptop (Windows) (review of 18.11.2026).
    text = part[[features.LABEL, "water_last"]].to_csv(float_format="%.6f", lineterminator="\n")
    # skipna=False: one forecast that is not a number makes the MAE "not a number" too,
    # so G1 and G2 say no. A plain .mean() would quietly skip that day and compute the
    # MAE on the other days, and the gate could pass (found in a review on 17.11.2026).
    return {
        "validation_days": len(part),
        "validation_first": part.index.min().date().isoformat() if len(part) else "none",
        "validation_last": part.index.max().date().isoformat() if len(part) else "none",
        "validation_md5": hashlib.md5(text.encode("utf-8")).hexdigest(),
        "validation_mae": float((forecast - truth).abs().mean(skipna=False)),
        "validation_mae_persistence": float((baselines.persistence(part) - truth).abs().mean(skipna=False)),
        "validation_mae_row3": float((row3 - truth).abs().mean(skipna=False)),
    }


def check_free_forecasts(summary: dict) -> tuple[bool, str]:
    """G1: the procedure has a lower validation MAE than persistence and than row 3."""
    # Point estimates, no interval: on 274 validation days the intervals are too wide
    # to show a small gain (persistence: -0.062 to +0.002 degC on 17.11.2026). An
    # interval rule would reject every version; its purpose here is to find breakage,
    # and a broken procedure is far worse than these two free forecasts.
    numbers = [summary.get(key) for key in ("validation_mae", "validation_mae_persistence", "validation_mae_row3")]
    if summary.get("validation_days", 0) == 0 or not all(is_number(n) for n in numbers):
        return False, "failed: no validation MAE to compare"
    mae, persistence, row3 = numbers
    if mae < persistence and mae < row3:
        return True, f"passed: {mae:.4f} < persistence {persistence:.4f} and row 3 {row3:.4f}"
    return False, f"failed: {mae:.4f}, persistence {persistence:.4f}, row 3 {row3:.4f}"


def check_champion_validation(summary: dict, champion: dict | None) -> tuple[bool, str]:
    """G2: the validation MAE is at most the champion's + MAX_MAE_INCREASE, on the same validation days."""
    if champion is None:
        return True, "skipped: no champion yet"
    params, metrics = champion.get("params", {}), champion.get("metrics", {})
    # The champion's MAE is only comparable if its validation days and labels were the
    # same. A champion logged before this gate has no fingerprint: not comparable.
    if params.get("validation_md5") != summary.get("validation_md5"):
        return False, "failed: not comparable (other validation days or labels, or a champion from before the gate)"
    champion_mae = metrics.get("validation_mae")
    mae = summary.get("validation_mae")
    if not (is_number(champion_mae) and is_number(mae)):
        return False, "failed: a validation MAE is missing"
    if mae <= champion_mae + MAX_MAE_INCREASE:
        return True, f"passed: {mae:.4f} <= champion {champion_mae:.4f} + {MAX_MAE_INCREASE}"
    return False, f"failed: {mae:.4f} > champion {champion_mae:.4f} + {MAX_MAE_INCREASE}"


def check_shadow(candidate_forecast: pd.Series, champion_forecast: pd.Series | None) -> tuple[bool, str]:
    """G3: on the newest days the two models forecast at most MAX_SHADOW_DIFFERENCE apart."""
    if champion_forecast is None:
        return True, "skipped: no champion yet"
    # The same days on both sides, and real numbers only; anything else is "no".
    if len(candidate_forecast) == 0 or not candidate_forecast.index.equals(champion_forecast.index):
        return False, "failed: the two forecasts are not on the same days"
    difference = (candidate_forecast - champion_forecast).abs().to_numpy()
    if not np.isfinite(difference).all():
        return False, "failed: a forecast is not a number"
    largest = float(difference.max())
    if largest <= MAX_SHADOW_DIFFERENCE:
        return True, f"passed: largest difference {largest:.3f} degC on {len(difference)} days"
    return False, f"failed: largest difference {largest:.3f} degC > {MAX_SHADOW_DIFFERENCE}"


def check_plausible(candidate_forecast: pd.Series, newest: pd.DataFrame) -> tuple[bool, str]:
    """G4: on the newest days the predicted change is never bigger than MAX_CHANGE (in both directions)."""
    if len(candidate_forecast) == 0 or not candidate_forecast.index.equals(newest.index):
        return False, "failed: no forecast for the newest days"
    # The change the model predicts: the forecast minus the last reading.
    change = (candidate_forecast - newest["water_last"]).abs().to_numpy()
    if not np.isfinite(change).all():
        return False, "failed: a forecast is not a number"
    largest = float(change.max())
    if largest <= MAX_CHANGE:
        return True, f"passed: largest predicted change {largest:.3f} degC"
    return False, f"failed: predicted change {largest:.3f} degC > {MAX_CHANGE}"


def check_fresh_data(last_label: pd.Timestamp, table: pd.DataFrame, time: dt.datetime) -> tuple[bool, str]:
    """G5: at `time` (now), the row of the issue day is ready and the newest label is recent."""
    # Why here and not only in retrain.yml: the workflow's freshness step looks at the
    # features only, and a training started by hand (train.py on a laptop) skips the
    # workflow. This check is on the way of EVERY promotion (review of 18.11.2026).
    # Training and logging still work with old data (e.g. a reviewer who rebuilds an
    # old data version); only the promotion to champion is refused.
    if pd.isna(last_label):
        return False, "failed: no row with a label"
    # The same check as check_fresh.py at the end of the feature pipeline (MS2 review
    # point A2): the row of the issue day D, for the evening S of the newest daily job
    # that was due (19:00 UTC). Before, it was "the newest row with all features at most
    # 2 days old". On Monday (until 19:00 UTC) S is Sunday, and D is Sunday or, if
    # Sunday had too few readings before 18:00 when its job ran, Saturday.
    passed, _, text = check_fresh.check_issue_day(table, check_fresh.service_day(time))
    if not passed:
        return False, f"failed: {text}"
    # A label is the mean of the NEXT day, so it can only exist once that day has reached
    # its last 10-minute slot (23:50 Swiss time): the same rule as in features.py, from
    # the same function. A younger label means a broken table, and a model trained on it
    # would have learned from the future (review of 27.11.2026: on a Monday morning, a
    # label on Sunday's row passed, and so did a label from the future). The feature code
    # never writes such a label, so this only stops a broken table before it can become
    # the champion.
    ready = features.label_ready(last_label.normalize())
    if time < ready:
        return False, (
            f"failed: the newest label (row of {last_label:%d.%m.%Y}) is the mean of a day that "
            f"is not over yet (it can exist from {ready:%d.%m.%Y %H:%M} Swiss time)"
        )
    label_age = (time.astimezone(ZoneInfo(features.TIMEZONE)).date() - last_label.date()).days
    if label_age > MAX_LABEL_AGE_DAYS:
        return False, f"failed: the newest label is {label_age} days old (allowed: {MAX_LABEL_AGE_DAYS})"
    return True, f"passed: {text}; newest label {label_age} days old"


def run_checks(
    summary: dict,
    candidate: forecasting.TablePredictor,
    champion: dict | None,
    table: pd.DataFrame,
    last_label: pd.Timestamp,
) -> dict:
    """All five checks on the REGISTERED candidate (loaded back from MLflow). Returns check -> text."""
    # candidate: the model as it was loaded back from the registry, so the checks test
    # the artifact that would become the champion, not only the object in memory.
    # champion: None if it is confirmed that there is no champion; otherwise a dict with
    # its "model", "params" and "metrics" (track.champion_info).
    # table: the whole feature table. G3 and G4 use its newest rows with all features,
    # G5 the row of its issue day (which may miss a feature, so it is not in `newest`).
    # last_label: the date of the newest row with a label (the newest training row).
    newest = newest_rows(table)
    candidate_forecast = forecasting.temperature_for_rows(candidate, newest)
    champion_forecast = None
    if champion is not None:
        champion_forecast = forecasting.temperature_for_rows(champion["model"], newest)
    results = [
        check_free_forecasts(summary),
        check_champion_validation(summary, champion),
        check_shadow(candidate_forecast, champion_forecast),
        check_plausible(candidate_forecast, newest),
        check_fresh_data(last_label, table, now()),
    ]
    # strict=True: five names for five results, or an error (never a check dropped quietly).
    return {name: text for name, (_, text) in zip(CHECKS, results, strict=True)}


def passes(checks: dict) -> bool:
    """Promote only if every check passed or was skipped because there is no champion."""
    # A missing check counts as "no", so a forgotten check can never promote a model.
    return all(checks.get(name, "").startswith(("passed", "skipped")) for name in CHECKS)
