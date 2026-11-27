"""Tests for the MLflow tracking and the model registry. Everything stays in a temporary folder."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push.
# MLflow writes to a small database in tmp_path here, never to DagsHub.
import datetime as dt

import mlflow
import numpy as np
import pandas as pd
import pytest

from lakegeneva.common import feature_store, forecasting  # the read path; the shared step from the change to the temperature
from lakegeneva.common.features import LABEL  # the column with tomorrow's mean (the truth)
from lakegeneva.training import gate, track, train
from tests.test_train import fake_rows


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    """MLflow with its own empty database and folder, for one test."""
    monkeypatch.chdir(tmp_path)  # model files go to tmp_path/mlruns
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    yield mlflow.MlflowClient()
    mlflow.set_tracking_uri(None)


# A proven data version, as after `dvc pull` on GitHub: the DVC list confirms the table
# and a commit gives it back.
PROVEN = {"data_md5": "ab12cd.dir", "data_commit": "a" * 40, "data_tag": "none", "data_matches_dvc": "yes"}


@pytest.fixture
def trained(monkeypatch, tmp_path):
    """train.run() on a small fake table across TEST_START, with small settings."""
    monkeypatch.setattr(train, "MIN_TRAIN_ROWS", 20)
    monkeypatch.setattr(train, "ALPHAS", (0.1, 100.0))
    table = fake_rows(100, start="2024-12-01")
    # Like the real table: the newest two days have all features but no label yet.
    table.iloc[-2:, table.columns.get_loc(LABEL)] = np.nan
    # Like the real table: the readings before 18:00 of each day, all complete here, so
    # the newest day is the issue day D (features.issue_day).
    table["n_readings_d0"] = 108
    # "Now" for the freshness check G5: the evening of the newest row, after 18:00 (Swiss
    # time), as in a real run (the fake table is from 2024/2025, so the real clock would
    # make it look old).
    monkeypatch.setattr(gate, "now", lambda: newest_evening(table))
    # The table as a file, read through the feature store like in train.py, so the run
    # names the md5 of the table it read. tmp_path has no DVC list, so the version
    # comes out "unknown"; the tests use it as proven, like a run on GitHub after
    # dvc pull (test_an_unproven_data_version_is_logged_but_not_registered: unknown).
    path = tmp_path / "features.parquet"
    table.to_parquet(path)
    _, data = feature_store.load_features(path=path)
    data = data | PROVEN
    return train.run(table), train.model_rows(table), data, table


def newest_evening(table: pd.DataFrame, days_later: int = 0) -> dt.datetime:
    """19:30 UTC (after the 18:00 cutoff in Switzerland) on the newest day of the table, plus `days_later`."""
    day = table.index.max().date() + dt.timedelta(days=days_later)
    return dt.datetime.combine(day, dt.time(19, 30), tzinfo=dt.UTC)


def two_targets(all_days: float, big_change: float) -> pd.DataFrame:
    """An evaluation table with the model's MAE in both windows; targets 0.42 and 1.54."""
    return pd.DataFrame(
        {"model": [all_days, big_change], "target": [0.42, 1.54]},
        index=["test, all year", "test, change > 0.8"],
    )


def test_target_is_met_only_when_both_targets_are_met():
    assert track.meets_target(two_targets(0.40, 1.50))
    # The mixed results: one target met, the other missed. Neither counts as met.
    assert not track.meets_target(two_targets(0.40, 1.60))  # good overall, misses the big-change days
    assert not track.meets_target(two_targets(0.44, 1.50))  # good on big-change days, misses overall
    assert not track.meets_target(two_targets(0.44, 1.60))


def test_target_is_not_met_when_a_window_was_not_tested():
    # Only the all-days row: the big-change target cannot be judged, so it is not met.
    only_all_days = pd.DataFrame({"model": [0.40], "target": [0.42]}, index=["test, all year"])
    assert not track.meets_target(only_all_days)
    # A window without an MAE (no big-change test days) is not met either.
    assert not track.meets_target(two_targets(0.40, float("nan")))


def test_flat_metrics_names_every_number_and_skips_empty_ones():
    evaluation = pd.DataFrame(
        {"days": [100, 5], "events": [np.nan, 3], "model": [0.4, 1.2], "ci_high": [-0.1, np.nan]},
        index=["test, all year", "test, change > 0.8"],
    )

    metrics = track.flat_metrics(evaluation)

    assert metrics == {
        "test_days": 100,
        "test_mae": 0.4,
        "test_ci_high": -0.1,
        "big_change_days": 5,
        "big_change_events": 3,  # runs of consecutive big-change days (review point B3)
        "big_change_mae": 1.2,
    }


def test_a_run_logs_both_models_and_registers_the_full_one(local_mlflow, trained):
    results, rows, data, table = trained

    track.track(results, rows, data, table)

    runs = mlflow.search_runs(experiment_names=[track.EXPERIMENT])
    assert sorted(runs["tags.mlflow.runName"]) == ["full", "water_only"]
    full = runs[runs["tags.mlflow.runName"] == "full"].iloc[0]
    # The code version, the data version and the test result are in every run.
    assert full["params.git_commit"] != ""
    assert full["params.data_md5"] != ""
    assert full["metrics.test_mae"] == pytest.approx(results["full"]["evaluation"].loc["test, all year", "model"])
    # Only the full model is registered: one version.
    versions = local_mlflow.search_model_versions(f"name='{track.MODEL_NAME}'")
    assert len(versions) == 1


def test_an_unproven_data_version_is_logged_but_not_registered(local_mlflow, trained):
    results, rows, data, table = trained
    # The table cannot be matched to a DVC version (no DVC list on this computer).
    unproven = data | {"data_md5": "unknown", "data_commit": "unknown", "data_tag": "unknown", "data_matches_dvc": "unknown"}

    decision = track.track(results, rows, unproven, table)

    # Both runs are logged, with the data version as it is ...
    runs = mlflow.search_runs(experiment_names=[track.EXPERIMENT])
    assert len(runs) == 2
    assert set(runs["params.data_matches_dvc"]) == {"unknown"}
    # ... but no model is registered, so no registered model points to an unknown version.
    assert local_mlflow.search_model_versions(f"name='{track.MODEL_NAME}'") == []
    assert decision["version"] is None and not decision["promoted"]
    assert "data version not proven" in decision["error"]


def test_the_champion_is_the_full_model_and_loads_back(local_mlflow, trained):
    results, rows, data, table = trained

    # The first run: no champion yet, and the fake data is easy to learn, so the full
    # model passes the gate and becomes the champion.
    decision = track.track(results, rows, data, table)
    assert decision["promoted"], decision

    loaded = mlflow.sklearn.load_model(f"models:/{track.MODEL_NAME}@{track.CHAMPION}")
    x = rows[train.FEATURES_FULL].tail(5)
    # The live forecast will load exactly this model: the same predictions, the same columns.
    assert np.allclose(loaded.predict(x), results["full"]["model"].predict(x))
    assert list(loaded.feature_names_in_) == train.FEATURES_FULL


def test_the_champion_loaded_like_in_ms4_gives_a_temperature_not_a_change(local_mlflow, trained):
    results, rows, data, table = trained
    track.track(results, rows, data, table)

    # How the live forecast (MS4) will get its model: by the alias "champion" from the
    # registry. Then the shared function on the newest rows of the feature table.
    champion = mlflow.sklearn.load_model(f"models:/{track.MODEL_NAME}@{track.CHAMPION}")
    # The newest rows WITH a label, so that the forecast can be compared with the truth.
    labelled = rows.tail(5)
    forecast = forecasting.temperature_for_rows(champion, labelled)

    # The model's own answer is tomorrow's CHANGE: small numbers around 0, far away
    # from a water temperature (the fake last readings are between 8 and 21 degC).
    change = champion.predict(labelled[train.FEATURES_FULL])
    truth = labelled[LABEL]
    assert np.abs(change).max() < 3
    assert np.abs(change - truth).min() > 5
    # The shared function gives a TEMPERATURE: within 1 degC of the true mean of
    # tomorrow. (These rows were in the training data, so this says nothing about how
    # good the model is. It only shows that the number is a temperature.)
    assert (forecast - truth).abs().max() < 1
    # It is the last reading plus that change, one forecast per date ...
    assert np.allclose(forecast, labelled["water_last"] + change)
    assert forecast.index.equals(labelled.index)
    # ... and the same forecast as from the model before it was stored in MLflow.
    assert np.allclose(forecast, forecasting.temperature_for_rows(results["full"]["model"], labelled))


def test_the_registered_model_says_that_its_output_is_a_change(local_mlflow, trained):
    results, rows, data, table = trained
    track.track(results, rows, data, table)

    # The signature is what the model page in MLflow shows: the input columns and the
    # output. get_model_info reads it without loading the model.
    signature = mlflow.models.get_model_info(f"models:/{track.MODEL_NAME}@{track.CHAMPION}").signature

    # The output has a name that says it is a change in degC. Before, it had no name,
    # and a reader could take the number for the temperature.
    assert signature.outputs.input_names() == [forecasting.MODEL_OUTPUT]
    assert "change" in forecasting.MODEL_OUTPUT
    # The inputs are still the ten feature columns, in the order of training.
    assert signature.inputs.input_names() == train.FEATURES_FULL


def test_the_first_run_skips_the_champion_checks_and_the_second_compares(local_mlflow, trained):
    results, rows, data, table = trained

    first = track.track(results, rows, data, table)
    second = track.track(results, rows, data, table)

    # Version 1: no champion yet, so G2 and G3 are skipped; G1 and G4 ran.
    assert first["checks"]["g2_validation_vs_champion"].startswith("skipped")
    assert first["checks"]["g3_shadow_vs_champion"].startswith("skipped")
    assert first["checks"]["g1_validation_vs_free_forecasts"].startswith("passed")
    # Version 2: the same code and data, so the same validation days (G2 can compare)
    # and the same forecasts as the champion (G3: no difference at all).
    assert second["checks"]["g2_validation_vs_champion"].startswith("passed")
    assert second["checks"]["g3_shadow_vs_champion"] == "passed: largest difference 0.000 degC on 7 days"
    assert second["promoted"]
    assert local_mlflow.get_model_version_by_alias(track.MODEL_NAME, track.CHAMPION).version in (2, "2")
    # Each check is a tag of the version, so the model page shows why it passed.
    tags = local_mlflow.get_model_version(track.MODEL_NAME, "2").tags
    assert tags["gate"] == "passed"
    assert tags["gate_g3_shadow_vs_champion"].startswith("passed")
    # The validation days and their fingerprint are logged with the run (for the next G2).
    runs = mlflow.search_runs(experiment_names=[track.EXPERIMENT])
    assert runs["params.validation_md5"].notna().all()
    assert runs["metrics.validation_mae_row3"].notna().all()


def test_a_model_that_fails_the_gate_does_not_take_the_alias(local_mlflow, trained, monkeypatch):
    results, rows, data, table = trained
    track.track(results, rows, data, table)  # version 1 passes and becomes the champion

    # The next run fails the shadow check (here: no difference at all is allowed any
    # more, a limit below 0). It is registered as version 2, but the alias stays.
    monkeypatch.setattr(gate, "MAX_SHADOW_DIFFERENCE", -1.0)
    decision = track.track(results, rows, data, table)

    assert not decision["promoted"]
    champion = local_mlflow.get_model_version_by_alias(track.MODEL_NAME, track.CHAMPION)
    assert champion.version in (1, "1")
    failed = local_mlflow.get_model_version(track.MODEL_NAME, "2")
    assert failed.tags["gate"] == "failed"
    assert failed.tags["gate_g3_shadow_vs_champion"].startswith("failed")


def test_a_champion_that_cannot_be_loaded_is_an_error_not_a_pass(local_mlflow, trained, monkeypatch):
    results, rows, data, table = trained
    track.track(results, rows, data, table)  # version 1 becomes the champion

    # The champion exists, but it cannot be read (e.g. DagsHub does not answer). This is
    # NOT the same as "no champion": the checks must not be skipped (fail-closed).
    def broken():
        raise ConnectionError("DagsHub does not answer")

    monkeypatch.setattr(track, "champion_info", broken)
    decision = track.track(results, rows, data, table)

    assert not decision["promoted"]
    assert "ConnectionError" in decision["error"]
    tags = local_mlflow.get_model_version(track.MODEL_NAME, "2").tags
    assert tags["gate"] == "error"
    assert local_mlflow.get_model_version_by_alias(track.MODEL_NAME, track.CHAMPION).version in (1, "1")


def test_the_champion_is_loaded_by_the_version_whose_metrics_were_read(local_mlflow, trained, monkeypatch):
    results, rows, data, table = trained
    track.track(results, rows, data, table)  # version 1 becomes the champion

    # Remember which model address champion_info() loads.
    loaded = []
    real_load = mlflow.sklearn.load_model
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: loaded.append(uri) or real_load(uri))

    champion = track.champion_info()

    # By the version number, not by "@champion": if the alias moved in between, the model
    # and the metrics would come from two different versions (review of 17.11.2026).
    assert loaded == [f"models:/{track.MODEL_NAME}/{champion['version']}"]


def test_a_rollback_while_the_gate_runs_is_not_undone(local_mlflow, trained, monkeypatch):
    results, rows, data, table = trained
    track.track(results, rows, data, table)  # version 1: champion
    track.track(results, rows, data, table)  # version 2: champion

    # While the gate of version 3 runs (after its checks against version 2), a person
    # rolls back to version 1 by hand.
    real_run_checks = gate.run_checks

    def checks_then_rollback(*args, **kwargs):
        checks = real_run_checks(*args, **kwargs)
        track.promote_by_hand("1")
        return checks

    monkeypatch.setattr(gate, "run_checks", checks_then_rollback)
    decision = track.track(results, rows, data, table)

    # Version 3 was checked against version 2, not against the champion of now: it must
    # not take the alias, or the retrain would undo the rollback (review of 18.11.2026).
    assert not decision["promoted"]
    assert "champion changed" in decision["error"]
    assert local_mlflow.get_model_version_by_alias(track.MODEL_NAME, track.CHAMPION).version in (1, "1")
    assert local_mlflow.get_model_version(track.MODEL_NAME, "3").tags["gate"] == "error"


def test_old_data_stops_the_promotion_but_not_the_logging(local_mlflow, trained, monkeypatch):
    results, rows, data, table = trained
    # 60 days later, without any new row: the table stopped growing.
    monkeypatch.setattr(gate, "now", lambda: newest_evening(table, days_later=60))

    decision = track.track(results, rows, data, table)

    # The run is logged and the version registered (old data can still be evaluated),
    # but it does not become the champion.
    assert not decision["promoted"]
    assert decision["checks"]["g5_fresh_data"].startswith("failed")
    assert len(mlflow.search_runs(experiment_names=[track.EXPERIMENT])) == 2


def test_a_person_can_promote_a_version_by_hand(local_mlflow, trained, monkeypatch):
    results, rows, data, table = trained
    track.track(results, rows, data, table)  # version 1: champion
    monkeypatch.setattr(gate, "MAX_SHADOW_DIFFERENCE", -1.0)
    track.track(results, rows, data, table)  # version 2: not promoted by the gate

    # A person checked version 2 and promotes it (when the gate could not decide).
    track.promote_by_hand("2")

    assert local_mlflow.get_model_version_by_alias(track.MODEL_NAME, track.CHAMPION).version in (2, "2")
    # The tag says that a person decided, not the gate.
    assert local_mlflow.get_model_version(track.MODEL_NAME, "2").tags["gate"] == "promoted by hand"
