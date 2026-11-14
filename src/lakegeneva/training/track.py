"""Experiment tracking and model registry with MLflow (hosted on DagsHub).

Every training run is logged with the code version (git commit), the data version
(DVC), its settings, its test results and the trained model. The full model is
registered as a new version of "lake-geneva-temp", and the alias "champion" moves to
it only if it passes the promotion gate (see passes_gate below).
"""
# Based on the MLOps HS26 Block 1 exercise solution (mlops-quotes, train.py), F. Bär
# [5] in README.md: log the parameters, the git commit, the metrics and the model,
# register every model, and move the alias "champion" to the one that is used.
# Changed for this project: the data version from DVC is logged too, the evaluation
# tables are saved with the run, and a gate decides about the alias.
#
# Where the runs go: to the address in the environment variable MLFLOW_TRACKING_URI
# (DagsHub: https://dagshub.com/<user>/lake-geneva-temp.mlflow, with the user name and
# a token in MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD). Without it, MLflow
# writes to a local database mlflow.db and the folder mlruns/ (both not in Git), which
# is how the code is tested on a laptop. `uv run mlflow ui` shows the local runs.
#
# MLflow is only in the dev group of pyproject.toml (like in the exercise): the Docker
# image of the feature pipeline does not need it. That is why it is imported inside the
# functions and not at the top of this file.

import subprocess  # runs git to find the code and data versions
from pathlib import Path  # the path of the DVC pointer file

import pandas as pd  # the evaluation tables

# The MLflow experiment (a folder of runs) and the registered model.
EXPERIMENT = "lake-geneva-temp"
MODEL_NAME = "lake-geneva-temp"
# The alias of the model that the live forecast will use (MS4).
CHAMPION = "champion"
# The DVC pointer of the feature table: its md5 is the version of the training data.
FEATURES_POINTER = Path("data/features.dvc")
# Short metric names for the windows of train.evaluate().
WINDOW_NAMES = {"test, all year": "test", "test, change > 0.8": "big_change", "test, Oct-Jan": "oct_jan"}
COLUMN_NAMES = {
    "days": "days",
    "model": "mae",
    "persistence": "mae_persistence",
    "target": "target",
    "gain_%": "gain_pct",
    "diff": "diff",
    "ci_low": "ci_low",
    "ci_high": "ci_high",
}


def git(*args: str) -> str:
    """The output of one git command, or "" if git is not available."""
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_commit() -> str:
    """The commit of the code that trained the model; "-dirty" if a tracked file was changed."""
    commit = git("rev-parse", "HEAD")
    if not commit:
        return "no-git"
    # --untracked-files=no: only changes to files that are in Git count. A run from
    # changed code cannot be repeated from the commit alone, so it gets a mark.
    changed = git("status", "--porcelain", "--untracked-files=no")
    return commit + ("-dirty" if changed else "")


def data_version(pointer: Path = FEATURES_POINTER) -> dict:
    """The version of the feature table: its DVC md5 and the commit that stored that pointer."""
    # The pointer file is short YAML; its line "- md5: <hash>.dir" names the data
    # version. `dvc checkout` after `git checkout <data_commit>` gives back this table.
    md5 = ""
    if pointer.exists():
        for line in pointer.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("- md5:"):
                md5 = line.split(":", 1)[1].strip()
    return {"data_md5": md5 or "unknown", "data_commit": git("log", "-1", "--format=%H", "--", str(pointer)) or "unknown"}


def flat_metrics(evaluation: pd.DataFrame) -> dict:
    """The evaluation table as single numbers for MLflow, e.g. test_mae or big_change_ci_high."""
    metrics = {}
    for window, prefix in WINDOW_NAMES.items():
        if window not in evaluation.index:
            continue
        for column, name in COLUMN_NAMES.items():
            value = evaluation.loc[window].get(column)
            if value is not None and pd.notna(value):  # windows without an interval are skipped
                metrics[f"{prefix}_{name}"] = float(value)
    return metrics


def passes_gate(evaluation: pd.DataFrame) -> bool:
    """The promotion gate: is the model clearly better than persistence on all test days?"""
    # "Clearly" means that the whole 95 % interval of (model - persistence) is below 0,
    # so the gain is not luck. A model that fails stays registered (for comparison), but
    # the alias "champion" keeps pointing at the last model that passed.
    ci_high = evaluation.loc["test, all year"].get("ci_high")
    return ci_high is not None and pd.notna(ci_high) and ci_high < 0


def log_run(name: str, result: dict, rows: pd.DataFrame, versions: dict) -> str:
    """Log one trained model with everything needed to repeat it; return its model URI."""
    import mlflow
    import mlflow.sklearn

    columns = result["columns"]
    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                "feature_set": name,
                "features": ",".join(columns),
                "alpha": result["alpha"],
                # What the model predicts; the forecast is the last reading plus this.
                "target": "tomorrow's mean minus the last reading before 18:00",
                "training_rows": len(rows),
                "first_row": rows.index.min().date().isoformat(),
                "last_row": rows.index.max().date().isoformat(),
                **versions,
            }
        )
        # Test results, and the validation error of every alpha that was tried.
        mlflow.log_metrics(flat_metrics(result["evaluation"]))
        mlflow.log_metrics({f"validation_mae_alpha_{alpha:g}": error for alpha, error in result["validation_mae"].items()})
        # The tables behind the numbers, to read or compare later.
        mlflow.log_text(result["evaluation"].to_csv(), "evaluation.csv")
        mlflow.log_text(result["forecast"].to_frame().join(rows[["water_last", "label_water_mean_d1"]]).to_csv(), "test_forecasts.csv")
        # The final model, trained on all rows. input_example: a few real rows, so the
        # model page shows which columns the model expects (its "signature").
        info = mlflow.sklearn.log_model(result["model"], name="model", input_example=rows[columns].head(3))
        return info.model_uri


def register(model_uri: str, promote: bool) -> str:
    """Register the model as a new version; move the alias "champion" to it if it passed the gate."""
    import mlflow

    # str(): MLflow 3 gives the version as a number; the registry calls take text too.
    version = str(mlflow.register_model(model_uri, MODEL_NAME).version)
    client = mlflow.MlflowClient()
    client.set_model_version_tag(MODEL_NAME, version, "gate", "passed" if promote else "failed")
    if promote:
        # Moving the alias is the promotion: the live forecast loads models:/<name>@champion.
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
    return version


def track(results: dict, rows: pd.DataFrame) -> None:
    """Log both models of train.run(); register the full model and promote it if it passes."""
    import mlflow

    mlflow.set_experiment(EXPERIMENT)
    versions = {"git_commit": git_commit(), **data_version()}
    for name, result in results.items():
        model_uri = log_run(name, result, rows, versions)
        # Only the full model is a candidate for the live forecast; the water-only model
        # is logged for the comparison (review point B2).
        if name == "full":
            promote = passes_gate(result["evaluation"])
            version = register(model_uri, promote)
            print(f"registered {MODEL_NAME} v{version}; gate {'passed: alias champion -> v' + version if promote else 'failed: champion unchanged'}")
    print(f"runs logged to {mlflow.get_tracking_uri()}")
