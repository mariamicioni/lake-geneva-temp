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

import hashlib  # md5 of the feature file that was really used
import json  # reads the DVC list of files of a data version
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
# Where DVC keeps its list of the files of each data version (default cache of DVC 3).
DVC_CACHE = Path(".dvc/cache/files/md5")
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


def file_md5(path: Path) -> str:
    """The md5 of a file's bytes (the same number DVC uses for a file)."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def data_version(features: Path, pointer: Path = FEATURES_POINTER, cache: Path = DVC_CACHE) -> dict:
    """Which data trained the model: the file really used, and its DVC version if it is one."""
    # Always true: the name and the md5 of the file that was read.
    features = Path(features)
    version = {"features_file": features.as_posix(), "features_md5": file_md5(features)}
    # The DVC version is claimed only if it is proven: the pointer names a list of
    # files (its line "- md5: <hash>.dir"), DVC keeps that list in its cache, and the
    # file used must be in that list with the same md5. Otherwise a run with another
    # table (--features) or a changed local table would name the wrong data version
    # (found in a review on 14.11.2026).
    dir_md5 = ""
    if pointer.exists():
        for line in pointer.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("- md5:"):
                dir_md5 = line.split(":", 1)[1].strip()
    listing = cache / dir_md5[:2] / dir_md5[2:] if dir_md5 else None
    if listing is None or not listing.exists():
        return version | {"data_md5": "unknown", "data_commit": "unknown", "data_matches_dvc": "unknown"}
    expected = {entry["relpath"]: entry["md5"] for entry in json.loads(listing.read_text(encoding="utf-8"))}
    if expected.get(features.name) != version["features_md5"]:
        return version | {"data_md5": "not a DVC version", "data_commit": "none", "data_matches_dvc": "no"}
    # `git checkout <data_commit>` and then `dvc checkout` give back exactly this table.
    data_commit = git("log", "-1", "--format=%H", "--", str(pointer)) or "unknown"
    # A pointer changed after its last commit is not yet a version anyone can check out.
    if git("status", "--porcelain", "--", str(pointer)):
        data_commit = "not committed"
    return version | {"data_md5": dir_md5, "data_commit": data_commit, "data_matches_dvc": "yes"}


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


def meets_target(evaluation: pd.DataFrame) -> bool:
    """The success criterion of the README: at least 20 % lower MAE than persistence on all test days."""
    row = evaluation.loc["test, all year"]
    return bool(row.get("model") <= row.get("target"))


def register(model_uri: str, promote: bool, target_met: bool = False) -> str:
    """Register the model as a new version; move the alias "champion" to it if it passed the gate."""
    import mlflow

    # str(): MLflow 3 gives the version as a number; the registry calls take text too.
    version = str(mlflow.register_model(model_uri, MODEL_NAME).version)
    client = mlflow.MlflowClient()
    client.set_model_version_tag(MODEL_NAME, version, "gate", "passed" if promote else "failed")
    # The gate is not the success criterion: a champion is clearly better than
    # persistence, but it may still miss the 20 % target. This tag says which.
    client.set_model_version_tag(MODEL_NAME, version, "target_20_percent", "met" if target_met else "not met")
    if promote:
        # Moving the alias is the promotion: the live forecast loads models:/<name>@champion.
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
    return version


def track(results: dict, rows: pd.DataFrame, features: Path) -> None:
    """Log both models of train.run(); register the full model and promote it if it passes."""
    import mlflow

    mlflow.set_experiment(EXPERIMENT)
    # features: the table train.py really read (--features), for the data version.
    versions = {"git_commit": git_commit(), **data_version(features)}
    for name, result in results.items():
        model_uri = log_run(name, result, rows, versions)
        # Only the full model is a candidate for the live forecast; the water-only model
        # is logged for the comparison (review point B2).
        if name == "full":
            promote = passes_gate(result["evaluation"])
            target_met = meets_target(result["evaluation"])
            version = register(model_uri, promote, target_met)
            print(f"registered {MODEL_NAME} v{version}; gate {'passed: alias champion -> v' + version if promote else 'failed: champion unchanged'}; 20 % target {'met' if target_met else 'not met'}")
    print(f"data: {versions['features_file']} (md5 {versions['features_md5'][:8]}), DVC version: {versions['data_matches_dvc']}")
    print(f"runs logged to {mlflow.get_tracking_uri()}")
