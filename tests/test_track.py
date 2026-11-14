"""Tests for the MLflow tracking and the model registry. Everything stays in a temporary folder."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push.
# MLflow writes to a small database in tmp_path here, never to DagsHub.
import mlflow
import numpy as np
import pandas as pd
import pytest

from lakegeneva.training import track, train
from tests.test_train import fake_rows


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    """MLflow with its own empty database and folder, for one test."""
    monkeypatch.chdir(tmp_path)  # model files go to tmp_path/mlruns
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    yield mlflow.MlflowClient()
    mlflow.set_tracking_uri(None)


@pytest.fixture
def trained(monkeypatch):
    """train.run() on a small fake table across TEST_START, with small settings."""
    monkeypatch.setattr(train, "MIN_TRAIN_ROWS", 20)
    monkeypatch.setattr(train, "ALPHAS", (0.1, 100.0))
    table = fake_rows(100, start="2024-12-01")
    return train.run(table), train.model_rows(table)


def evaluation_with_interval(ci_high: float) -> pd.DataFrame:
    """A minimal evaluation table: only what the gate reads."""
    return pd.DataFrame({"ci_high": [ci_high]}, index=["test, all year"])


def test_gate_promotes_only_a_model_that_is_clearly_better():
    # The whole interval of (model - persistence) below 0: clearly better.
    assert track.passes_gate(evaluation_with_interval(-0.06))
    # The interval reaches 0 or above: the gain could be luck.
    assert not track.passes_gate(evaluation_with_interval(0.01))
    # No interval at all (too few test days): never promoted.
    assert not track.passes_gate(evaluation_with_interval(np.nan))


def test_data_version_reads_the_md5_of_the_dvc_pointer(tmp_path):
    pointer = tmp_path / "features.dvc"
    pointer.write_text("outs:\n- md5: abc123.dir\n  size: 1\n  path: features\n", encoding="utf-8")

    assert track.data_version(pointer)["data_md5"] == "abc123.dir"
    # A missing pointer does not stop training; the run says the version is unknown.
    assert track.data_version(tmp_path / "missing.dvc")["data_md5"] == "unknown"


def test_flat_metrics_names_every_number_and_skips_empty_ones():
    evaluation = pd.DataFrame(
        {"days": [100, 5], "model": [0.4, 1.2], "ci_high": [-0.1, np.nan]},
        index=["test, all year", "test, change > 0.8"],
    )

    metrics = track.flat_metrics(evaluation)

    assert metrics == {"test_days": 100, "test_mae": 0.4, "test_ci_high": -0.1, "big_change_days": 5, "big_change_mae": 1.2}


def test_a_run_logs_both_models_and_registers_the_full_one(local_mlflow, trained):
    results, rows = trained

    track.track(results, rows)

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


def test_the_champion_is_the_full_model_and_loads_back(local_mlflow, trained):
    results, rows = trained
    # The fake data is easy to learn, so the full model passes the gate.
    assert track.passes_gate(results["full"]["evaluation"])

    track.track(results, rows)

    loaded = mlflow.sklearn.load_model(f"models:/{track.MODEL_NAME}@{track.CHAMPION}")
    x = rows[train.FEATURES_FULL].tail(5)
    # The live forecast will load exactly this model: the same predictions, the same columns.
    assert np.allclose(loaded.predict(x), results["full"]["model"].predict(x))
    assert list(loaded.feature_names_in_) == train.FEATURES_FULL


def test_a_model_that_fails_the_gate_does_not_take_the_alias(local_mlflow, trained, monkeypatch):
    results, rows = trained
    track.track(results, rows)  # version 1 passes and becomes the champion

    # The next run fails the gate: it is registered as version 2, but the alias stays.
    monkeypatch.setattr(track, "passes_gate", lambda evaluation: False)
    track.track(results, rows)

    champion = local_mlflow.get_model_version_by_alias(track.MODEL_NAME, track.CHAMPION)
    assert champion.version in (1, "1")
    failed = local_mlflow.get_model_version(track.MODEL_NAME, "2")
    assert failed.tags["gate"] == "failed"
